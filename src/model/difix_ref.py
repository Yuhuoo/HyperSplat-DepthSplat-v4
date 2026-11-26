from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class DifixRefCfg:
    enabled: bool = False
    model_path: str = "nvidia/difix_ref"
    repo_path: Path = Path("../Difix3D")
    local_files_only: bool = False
    fp16: bool = True
    max_side: int | None = None


def select_reference_views(
    context_poses: Tensor,
    target_poses: Tensor,
    context_indices: Tensor,
    target_indices: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return each target's closest other context view and a mask of novel targets."""
    context_poses = context_poses.float()
    target_poses = target_poses.float()
    translation = torch.cdist(target_poses[:, :3, 3], context_poses[:, :3, 3])
    # The trace of the relative rotation gives its angle in radians.
    target_rotation = target_poses[:, :3, :3].flatten(1)
    context_rotation = context_poses[:, :3, :3].flatten(1)
    trace = target_rotation @ context_rotation.T
    rotation = ((trace - 1) / 2).clamp(-1, 1).acos()
    same_frame = target_indices[:, None] == context_indices[None, :]
    if same_frame.all(dim=1).any():
        raise ValueError("Difix-ref needs a context frame distinct from each target.")
    distances = (translation + rotation).masked_fill(same_frame, torch.inf)
    return distances.argmin(dim=1), ~same_frame.any(dim=1)


class DifixRefiner:
    def __init__(self, cfg: DifixRefCfg, device: torch.device, seed: int) -> None:
        if cfg.max_side is not None and cfg.max_side <= 0:
            raise ValueError("test.difix.max_side must be positive.")

        pipeline_file = cfg.repo_path.resolve() / "src" / "pipeline_difix.py"
        if not pipeline_file.is_file():
            raise FileNotFoundError(f"Difix pipeline not found at {pipeline_file}. Set test.difix.repo_path.")

        spec = spec_from_file_location("depthsplat_difix_pipeline", pipeline_file)
        pipeline_module = module_from_spec(spec)
        try:
            spec.loader.exec_module(pipeline_module)
        except ImportError as exc:
            raise ImportError("Install the optional dependencies from requirements-difix.txt.") from exc

        self.pipeline = pipeline_module.DifixPipeline.from_pretrained(
            cfg.model_path,
            torch_dtype=torch.float16 if cfg.fp16 else torch.float32,
            local_files_only=cfg.local_files_only,
            trust_remote_code=True,
        ).to(device)
        self.pipeline.set_progress_bar_config(disable=True)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.max_side = cfg.max_side

    @torch.inference_mode()
    def __call__(self, renders: Tensor, references: Tensor, reference_indices: Tensor) -> Tensor:
        height, width = renders.shape[-2:]
        scale = min(1.0, self.max_side / max(height, width)) if self.max_side else 1.0
        multiple = self.pipeline.vae_scale_factor
        size = tuple(max(multiple, int(dim * scale) // multiple * multiple) for dim in (height, width))

        def prepare(images: Tensor) -> Tensor:
            images = images.float().clamp(0, 1)
            if images.shape[-2:] != size:
                images = F.interpolate(images, size=size, mode="bilinear", align_corners=False, antialias=True)
            return images

        references = prepare(references)
        repaired = torch.empty_like(renders, dtype=torch.float32)
        # Each call supplies exactly one render/reference pair to the reference UNet.
        for i, ref_idx in enumerate(reference_indices.tolist()):
            image = self.pipeline(
                prompt="remove degradation",
                image=prepare(renders[i : i + 1]),
                ref_image=references[ref_idx : ref_idx + 1],
                num_inference_steps=1,
                timesteps=[199],
                guidance_scale=0.0,
                output_type="pt",
                generator=self.generator,
            ).images.float()
            if image.shape[-2:] != (height, width):
                image = F.interpolate(image, size=(height, width), mode="bilinear", align_corners=False)
            repaired[i] = image[0].clamp(0, 1)
        if not torch.isfinite(repaired).all():
            raise RuntimeError("Difix-ref produced non-finite pixels. Try test.difix.fp16=false.")
        return repaired
