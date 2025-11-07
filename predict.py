import atexit
import importlib
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from urllib.parse import urlparse
from urllib.request import urlopen

import mediapy
import torch
import torch.distributed as dist
import torchvision.transforms as T
from cog import BasePredictor, Input, Path as CogPath
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torchvision.io import read_video
from torchvision.transforms import Compose, Lambda, Normalize

from common.config import load_config
from common.distributed import init_torch
from common.distributed.ops import sync_data
from common.seed import set_seed
from data.image.transforms.divisible_crop import DivisibleCrop
from data.image.transforms.na_resize import NaResize
from data.video.transforms.rearrange import Rearrange
from projects.video_diffusion_sr.infer import VideoDiffusionInfer

MODEL_CACHE = Path("model_cache")
BASE_URL = "https://weights.replicate.delivery/default/seedvr2/model_cache/"
CKPT_DIR = MODEL_CACHE / "weights"
WHEEL_DIR = MODEL_CACHE / "wheels"

os.environ.setdefault("HF_HOME", str(MODEL_CACHE))
os.environ.setdefault("TORCH_HOME", str(MODEL_CACHE))
os.environ.setdefault("HF_DATASETS_CACHE", str(MODEL_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODEL_CACHE))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_CACHE.mkdir(parents=True, exist_ok=True)

SHARED_WEIGHT_FILES = {
    "vae": "ema_vae.pth",
    "pos_emb": "pos_emb.pt",
    "neg_emb": "neg_emb.pt",
}

# Runner metadata

@dataclass(frozen=True)
class ModelSpec:
    weight: str
    config: str


MODEL_VARIANTS: Dict[str, ModelSpec] = {
    "3b": ModelSpec(weight="seedvr2_ema_3b.pth", config="configs_3b/main.yaml"),
    "7b": ModelSpec(weight="seedvr2_ema_7b.pth", config="configs_7b/main.yaml"),
}
DEFAULT_MODEL_VARIANT = "3b"

APEX_WHEEL = "apex-0.1-cp310-cp310-linux_x86_64.whl"
FLASH_ATTN_SPEC = "flash_attn"
MODEL_FILES = [
    ".cache.tar",
    "version.txt",
    "version_diffusers_cache.txt",
    "weights.tar",
    "wheels.tar",
    "xet.tar",
]


def download_weights(url: str, dest: Path) -> None:
    start = time.time()
    print("[!] Initiating download from URL:", url)
    print("[~] Destination path:", dest)
    target = dest if dest.suffix != ".tar" else dest.parent
    command = ["pget", "-vf" + ("x" if dest.suffix == ".tar" else ""), url, str(target)]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in:", time.time() - start, "seconds")


def _resolve_numeric(value: Union[int, float], default: Union[int, float]) -> Union[int, float]:
    if isinstance(value, (int, float)):
        return value
    return default


def ensure_weight(filename: str) -> Path:
    path = CKPT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Expected weight {path} missing. Did the CDN download succeed?")
    return path


def cut_videos(video: torch.Tensor, sp_size: int) -> torch.Tensor:
    if video.size(1) > 121:
        video = video[:, :121]
    t = video.size(1)
    if t <= 4 * sp_size:
        padding = [video[:, -1].unsqueeze(1)] * (4 * sp_size - t + 1)
        padding = torch.cat(padding, dim=1)
        return torch.cat([video, padding], dim=1)
    if (t - 1) % (4 * sp_size) == 0:
        return video
    padding = [video[:, -1].unsqueeze(1)] * (4 * sp_size - ((t - 1) % (4 * sp_size)))
    padding = torch.cat(padding, dim=1)
    video = torch.cat([video, padding], dim=1)
    return video


def mux_audio_stream(src_media: Path, video_only: Path, output_path: Path) -> Path:
    """Copy the source audio stream onto `video_only` without re-encoding."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        print("[WARN] ffmpeg missing; returning video without audio passthrough.")
        video_only.replace(output_path)
        return output_path

    ffmpeg_cmd = [
        ffmpeg_path,
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_only),
        "-i",
        str(src_media),
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-shortest",
        str(output_path),
    ]

    result = subprocess.run(ffmpeg_cmd, check=False, capture_output=True)
    if result.returncode == 0:
        video_only.unlink(missing_ok=True)
        return output_path

    stderr = result.stderr.decode("utf-8", errors="ignore") if result.stderr else ""
    print("[WARN] Audio mux failed, returning video-only result.")
    if stderr:
        print(stderr)
    video_only.replace(output_path)
    return output_path


def ensure_model_cache() -> None:
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    for model_file in MODEL_FILES:
        url = BASE_URL + model_file
        dest_path = MODEL_CACHE / model_file
        if model_file.endswith(".tar"):
            extracted_path = dest_path.parent / model_file.replace(".tar", "")
            if extracted_path.exists():
                continue
        elif dest_path.exists():
            continue
        download_weights(url, dest_path)


class LazyRunnerManager:
    def __init__(self, device: torch.device, builder) -> None:
        self._device = device
        self._builder = builder
        self._bundles: Dict[str, Dict] = {}
        self._active: Optional[str] = None

    def use(self, variant: str) -> Tuple["VideoDiffusionInfer", OmegaConf]:
        bundle = self._bundles.get(variant)
        previous = self._active if self._active in self._bundles else None

        if bundle is None:
            if previous is not None:
                self._move_to_cpu(self._bundles[previous])
                torch.cuda.empty_cache()
            runner, config = self._builder(variant)
            bundle = {"runner": runner, "config": config, "device": "cuda"}
            self._bundles[variant] = bundle
        else:
            if previous is not None and previous != variant:
                self._move_to_cpu(self._bundles[previous])
                torch.cuda.empty_cache()
            if bundle["device"] != "cuda":
                self._move_to_cuda(bundle)

        self._active = variant
        return bundle["runner"], bundle["config"]

    def _move_to_cuda(self, bundle: Dict) -> None:
        runner: VideoDiffusionInfer = bundle["runner"]
        runner.dit.to(self._device)
        runner.vae.to(self._device)
        runner.device = "cuda"
        if hasattr(runner.vae, "set_memory_limit"):
            runner.vae.set_memory_limit(**runner.config.vae.memory_limit)
        bundle["device"] = "cuda"

    @staticmethod
    def _move_to_cpu(bundle: Dict) -> None:
        if bundle["device"] == "cpu":
            return
        runner: VideoDiffusionInfer = bundle["runner"]
        runner.dit.to("cpu")
        runner.vae.to("cpu")
        runner.device = "cpu"
        bundle["device"] = "cpu"


class Predictor(BasePredictor):
    def setup(self) -> None:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.pop("TRANSFORMERS_CACHE", None)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.device = torch.device("cuda")

        ensure_model_cache()
        self._ensure_flash_attn()
        self._ensure_apex()

        shared = {name: ensure_weight(path) for name, path in SHARED_WEIGHT_FILES.items()}
        self.pos_emb = torch.load(shared["pos_emb"], map_location=self.device, weights_only=True)
        self.neg_emb = torch.load(shared["neg_emb"], map_location=self.device, weights_only=True)

        init_torch(cudnn_benchmark=False)

        self.dual_load = self._should_preload_all_variants()
        if self.dual_load:
            self.runners: Dict[str, VideoDiffusionInfer] = {}
            self.configs: Dict[str, OmegaConf] = {}
            for variant in MODEL_VARIANTS:
                runner, config = self._build_runner(variant)
                self.runners[variant] = runner
                self.configs[variant] = config
        else:
            self.lazy_runners = LazyRunnerManager(self.device, self._build_runner)
            self.lazy_runners.use(DEFAULT_MODEL_VARIANT)

        if not getattr(self, "_destroy_registered", False):
            atexit.register(self._maybe_destroy_pg)
            self._destroy_registered = True

        # cache frequently used transforms
        self.video_transform = Compose(
            [
                NaResize(resolution=(1280 * 720) ** 0.5, mode="area", downsample_only=False),
                Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
                DivisibleCrop((16, 16)),
                Normalize(0.5, 0.5),
                Rearrange("t c h w -> c t h w"),
            ]
        )
        self.image_transform = Compose(
            [
                NaResize(resolution=(2560 * 1440) ** 0.5, mode="area", downsample_only=False),
                Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
                DivisibleCrop((16, 16)),
                Normalize(0.5, 0.5),
                Rearrange("t c h w -> c t h w"),
            ]
        )

    def predict(
        self,
        media: CogPath = Input(
            description="Video (mp4/mov) or image (png/jpg/webp) to restore.",
        ),
        cfg_scale: float = Input(
            description="Classifier-free guidance scale (higher = stronger restoration).",
            default=1.0,
            ge=0.0,
            le=15.0,
        ),
        sample_steps: int = Input(
            description="Sampling steps (1 = fast one-step mode).",
            default=1,
            ge=1,
            le=4,
        ),
        sp_size: int = Input(
            description="Sequence-parallel shard count. Increase for long clips.",
            default=1,
            ge=1,
            le=4,
        ),
        fps: int = Input(
            description="Frames-per-second for video outputs.",
            default=24,
            ge=1,
            le=120,
        ),
        seed: Optional[int] = Input(
            description="Random seed. Leave blank for a random seed each call.",
            default=None,
        ),
        output_format: str = Input(
            description="Image output format (only used for image inputs).",
            choices=["png", "webp", "jpg"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Image quality for lossy formats (jpg/webp).",
            default=90,
            ge=10,
            le=100,
        ),
        model_variant: str = Input(
            description="Model size to run.",
            choices=list(MODEL_VARIANTS.keys()),
            default=DEFAULT_MODEL_VARIANT,
        ),
    ) -> CogPath:
        input_path, cleanup = self._resolve_media_path(media)
        media_kind = self._detect_media_kind(input_path)
        if media_kind not in {"image", "video"}:
            raise ValueError("Unsupported file type. Provide a video or image.")

        cfg_scale_val = float(_resolve_numeric(cfg_scale, 1.0))
        sample_steps_val = int(_resolve_numeric(sample_steps, 1))
        fps_val = int(_resolve_numeric(fps, 24))
        sp_size_val = int(_resolve_numeric(sp_size, 1))
        seed_val = int(_resolve_numeric(seed if seed is not None else torch.randint(0, 2**32, ()).item(), 666))

        if model_variant not in MODEL_VARIANTS:
            raise ValueError(f"Unknown model variant '{model_variant}'. Choose from {list(MODEL_VARIANTS.keys())}.")

        if self.dual_load:
            runner = self.runners[model_variant]
            config = self.configs[model_variant]
        else:
            runner, config = self.lazy_runners.use(model_variant)
        config.diffusion.cfg.scale = cfg_scale_val
        config.diffusion.cfg.rescale = 0.0
        config.diffusion.timesteps.sampling.steps = sample_steps_val
        runner.configure_diffusion()

        set_seed(seed_val, same_across_ranks=True)

        text_embeds = {
            "texts_pos": [self.pos_emb.clone()],
            "texts_neg": [self.neg_emb.clone()],
        }

        # single item list, process directly
        cond_latents = []
        ori_lengths = []

        if media_kind == "video":
            frames, _, _ = read_video(str(input_path), output_format="TCHW", pts_unit="sec")
            frames = frames.float() / 255.0
            if frames.size(0) > 121:
                frames = frames[:121]
            tensor = frames.to(self.device)
            cond = self.video_transform(tensor)
        else:
            image = Image.open(str(input_path)).convert("RGB")
            img_tensor = T.ToTensor()(image).unsqueeze(0)
            tensor = img_tensor.to(self.device)
            cond = self.image_transform(tensor)

        cond_latents.append(cond)
        ori_lengths.append(cond.size(1))

        if media_kind == "video":
            cond_latents = [cut_videos(cond, sp_size_val) for cond in cond_latents]

        with torch.inference_mode():
            cond_latents = runner.vae_encode(cond_latents)
            samples = self._generation_step(runner, cond_latents, text_embeds)

        sample = samples[0]
        if ori_lengths[0] < sample.shape[0]:
            sample = sample[: ori_lengths[0]]

        sample = rearrange(sample[:, None], "t c h w -> t h w c") if sample.ndim == 3 else rearrange(sample, "t c h w -> t h w c")
        sample = sample.clip(-1, 1).mul_(0.5).add_(0.5).mul_(255).round().to(torch.uint8).cpu().numpy()

        if media_kind == "image":
            img_array = sample[0]
            target_ext = output_format.lower()
            encoded_ext = "jpeg" if target_ext == "jpg" else target_ext
            output_name = Path(f"output_{uuid.uuid4().hex}.{target_ext}")
            pil_img = Image.fromarray(img_array)
            save_kwargs = {}
            if encoded_ext in {"jpeg", "webp"}:
                save_kwargs["quality"] = output_quality
                save_kwargs["optimize"] = True
            pil_img.save(str(output_name), format=encoded_ext.upper(), **save_kwargs)
        else:
            final_video = Path(f"output_{uuid.uuid4().hex}.mp4")
            temp_video = final_video.with_name(f"{final_video.stem}_video.mp4")
            mediapy.write_video(str(temp_video), sample, fps=fps_val)
            output_name = mux_audio_stream(input_path, temp_video, final_video)

        torch.cuda.empty_cache()
        if cleanup:
            input_path.unlink(missing_ok=True)
        return CogPath(str(output_name))

    def _build_runner(self, variant: str) -> Tuple["VideoDiffusionInfer", OmegaConf]:
        spec = MODEL_VARIANTS.get(variant)
        if spec is None:
            raise ValueError(f"Unknown model variant '{variant}'. Choose from {list(MODEL_VARIANTS.keys())}.")
        weight_path = ensure_weight(spec.weight)
        config = load_config(spec.config)
        OmegaConf.set_readonly(config, False)

        dit_model = config.dit.model
        dit_model.norm = "rms"
        dit_model.vid_out_norm = "rms"
        if hasattr(dit_model, "txt_in_norm"):
            dit_model.txt_in_norm = "layer"
        if hasattr(dit_model, "qk_norm"):
            dit_model.qk_norm = "rms"

        runner = VideoDiffusionInfer(config)
        runner.configure_dit_model(device="cuda", checkpoint=str(weight_path))
        runner.configure_vae_model()
        if hasattr(runner.vae, "set_memory_limit"):
            runner.vae.set_memory_limit(**runner.config.vae.memory_limit)

        return runner, config

    def _should_preload_all_variants(self) -> bool:
        env_force_stage = os.getenv("SEEDVR_FORCE_STAGE", "").lower()
        if env_force_stage in {"1", "true", "yes"}:
            return False

        env_force_dual = os.getenv("SEEDVR_FORCE_DUAL_LOAD", "").lower()
        if env_force_dual in {"1", "true", "yes"}:
            return True

        try:
            total_mem = torch.cuda.get_device_properties(0).total_memory
        except (RuntimeError, AttributeError):
            return False

        return total_mem >= 120 * 1024**3

    @staticmethod
    def _resolve_media_path(media: Union[str, Path]) -> Tuple[Path, bool]:
        candidate = Path(media)
        if candidate.exists():
            return candidate, False

        media_str = str(media)
        parsed = urlparse(media_str)
        if parsed.scheme in {"http", "https"}:
            suffix = Path(parsed.path).suffix or ""
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".bin") as tmp:
                with urlopen(media_str) as src, open(tmp.name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            return Path(tmp.name), True

        raise ValueError("Provide a local file path or an HTTP(S) URL.")

    @staticmethod
    def _detect_media_kind(path: Path) -> Optional[str]:
        mime, _ = mimetypes.guess_type(str(path))
        if mime:
            if mime.startswith("image"):
                return "image"
            if mime.startswith("video"):
                return "video"

        # fallback: try to open as image
        try:
            with Image.open(str(path)) as img:
                img.verify()
            return "image"
        except Exception:
            pass

        return "video"

    def _generation_step(self, runner: "VideoDiffusionInfer", cond_latents, text_embeds):
        noises = [torch.randn_like(latent) for latent in cond_latents]
        aug_noises = [torch.randn_like(latent) for latent in cond_latents]
        noises, aug_noises, cond_latents = sync_data((noises, aug_noises, cond_latents), 0)

        noises = [noise.to(self.device) for noise in noises]
        aug_noises = [noise.to(self.device) for noise in aug_noises]
        cond_latents = [latent.to(self.device) for latent in cond_latents]

        cond_noise_scale = 0.1

        def _add_noise(x, aug_noise):
            t = torch.tensor([1000.0], device=self.device) * cond_noise_scale
            shape = torch.tensor(x.shape[1:], device=self.device)[None]
            t = runner.timestep_transform(t, shape)
            return runner.schedule.forward(x, aug_noise, t)

        conditions = [
            runner.get_condition(noise, task="sr", latent_blur=_add_noise(latent_blur, aug_noise))
            for noise, aug_noise, latent_blur in zip(noises, aug_noises, cond_latents)
        ]

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            video_tensors = runner.inference(
                noises=noises,
                conditions=conditions,
                dit_offload=False,
                texts_pos=text_embeds["texts_pos"],
                texts_neg=text_embeds["texts_neg"],
            )

        samples = [
            rearrange(video[:, None], "c t h w -> t c h w") if video.ndim == 3 else rearrange(video, "c t h w -> t c h w")
            for video in video_tensors
        ]
        return samples

    @staticmethod
    def _maybe_destroy_pg():
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    @staticmethod
    def _ensure_package(module_name: str, install_args, extra_env=None) -> None:
        try:
            importlib.import_module(module_name)
            return
        except ImportError:
            pass

        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        cmd = [sys.executable, "-m", "pip", "install", *install_args]
        subprocess.run(cmd, check=True, env=env)

    def _ensure_flash_attn(self) -> None:
        self._ensure_package(
            module_name=FLASH_ATTN_SPEC,
            install_args=["--no-build-isolation", "flash-attn==2.5.9.post1"],
            extra_env={"FLASH_ATTENTION_SKIP_CUDA_BUILD": "TRUE"},
        )

    def _ensure_apex(self) -> None:
        try:
            importlib.import_module("apex.normalization.fused_layer_norm")
            return
        except ImportError:
            pass

        WHEEL_DIR.mkdir(parents=True, exist_ok=True)
        wheel_path = WHEEL_DIR / APEX_WHEEL
        if not wheel_path.exists():
            raise FileNotFoundError(
                f"Apex wheel not found at {wheel_path}. Ensure model cache download completed."
            )
        self._ensure_package(
            module_name="apex",
            install_args=[str(wheel_path)],
        )
