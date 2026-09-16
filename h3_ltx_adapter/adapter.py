"""Checkpoint validation and H3-to-LTX conversion."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from .constants import (
    H3_LATENT_CHANNELS,
    H3_LATENTS_MEAN,
    H3_LATENTS_STD,
    LTX_LATENT_CHANNELS,
)
from .geometry import (
    align_h3_to_ltx,
    h3_temporal_positions,
    ltx_temporal_positions,
    padded_ltx_pixel_frames,
    target_ltx_latent_frames,
)


@dataclass(frozen=True)
class AdapterConfig:
    path: Path
    payload: dict

    @property
    def model_config(self) -> dict:
        return self.payload["model_config"]


def load_adapter_config(path: str | Path) -> AdapterConfig:
    path = Path(path).resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "h3_ltx_adapter_safetensors_v1":
        raise ValueError(f"unsupported adapter format: {payload.get('format')!r}")
    expected_geometry = {
        "spatial_mode": "pixel_unshuffle",
        "temporal_mode": "linear_plus_nearest_pack",
        "temporal_pack_slots": 3,
    }
    if payload.get("geometry") != expected_geometry:
        raise ValueError(
            f"checkpoint geometry {payload.get('geometry')} != {expected_geometry}"
        )
    return AdapterConfig(path=path, payload=payload)


def normalize_raw_h3(raw_h3: torch.Tensor) -> torch.Tensor:
    mean = raw_h3.new_tensor(H3_LATENTS_MEAN).view(1, -1, 1, 1, 1)
    std = raw_h3.new_tensor(H3_LATENTS_STD).view(1, -1, 1, 1, 1)
    return (raw_h3 - mean) / std


def expected_shapes(
    pixel_frames: int, pixel_height: int, pixel_width: int
) -> dict[str, list[int]]:
    if pixel_height % 32 or pixel_width % 32:
        raise ValueError("pixel height and width must be divisible by 32")
    padded_frames = padded_ltx_pixel_frames(pixel_frames)
    return {
        "h3": [
            H3_LATENT_CHANNELS,
            len(h3_temporal_positions(pixel_frames)),
            pixel_height // 16,
            pixel_width // 16,
        ],
        "aligned_h3": [
            384,
            len(ltx_temporal_positions(padded_frames)),
            pixel_height // 32,
            pixel_width // 32,
        ],
        "ltx": [
            LTX_LATENT_CHANNELS,
            len(ltx_temporal_positions(padded_frames)),
            pixel_height // 32,
            pixel_width // 32,
        ],
    }


@torch.inference_mode()
def convert_h3_to_ltx(
    model: torch.nn.Module,
    h3_latent: torch.Tensor,
    *,
    pixel_frames: int,
    pixel_height: int,
    pixel_width: int,
    input_normalization: str,
    trim_to_source_duration: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if h3_latent.ndim == 4:
        h3_latent = h3_latent.unsqueeze(0)
    expected = expected_shapes(pixel_frames, pixel_height, pixel_width)
    if list(h3_latent.shape[1:]) != expected["h3"]:
        raise ValueError(
            f"H3 latent shape {list(h3_latent.shape[1:])} != "
            f"expected {expected['h3']}"
        )

    if input_normalization == "raw":
        h3_latent = normalize_raw_h3(h3_latent.float())
    elif input_normalization != "normalized":
        raise ValueError("input_normalization must be 'normalized' or 'raw'")

    aligned = align_h3_to_ltx(
        h3_latent.to(device=device, dtype=dtype),
        pixel_frames=pixel_frames,
        target_height=pixel_height // 32,
        target_width=pixel_width // 32,
        slots=3,
    )
    if list(aligned.shape[1:]) != expected["aligned_h3"]:
        raise RuntimeError(
            f"aligned H3 shape {list(aligned.shape[1:])} != "
            f"expected {expected['aligned_h3']}"
        )

    output = model(aligned)
    if list(output.shape[1:]) != expected["ltx"]:
        raise RuntimeError(
            f"LTX output shape {list(output.shape[1:])} != "
            f"expected {expected['ltx']}"
        )
    if not torch.isfinite(output).all():
        raise FloatingPointError("adapter produced NaN or Inf")

    if trim_to_source_duration:
        output = output[:, :, : target_ltx_latent_frames(pixel_frames)]
    return output.contiguous()
