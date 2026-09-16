"""ComfyUI nodes for the Efficient-Large-Model H3-to-LTX adapter."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path

import torch

import comfy.model_management as model_management
import comfy.model_patcher
import comfy.utils
import folder_paths

from .h3_ltx_adapter import (
    AdapterConfig,
    build_model,
    convert_h3_to_ltx,
    load_adapter_config,
)
from .h3_ltx_adapter.geometry import infer_h3_pixel_frames


LOGGER = logging.getLogger(__name__)
MODEL_FOLDER = "h3_ltx_adapters"
MODEL_ROOT = os.path.join(folder_paths.models_dir, MODEL_FOLDER)
folder_paths.add_model_folder_path(MODEL_FOLDER, MODEL_ROOT)
folder_paths.folder_names_and_paths[MODEL_FOLDER][1].update(
    folder_paths.supported_pt_extensions
)


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class LoadedH3ToLTXAdapter:
    model: torch.nn.Module
    patcher: comfy.model_patcher.ModelPatcher
    config: AdapterConfig
    dtype: torch.dtype
    model_path: Path


def _adapter_files() -> list[str]:
    return [
        name
        for name in folder_paths.get_filename_list(MODEL_FOLDER)
        if Path(name).name == "model.safetensors"
    ]


def _video_tensor(latent: dict) -> torch.Tensor:
    if not isinstance(latent, dict) or "samples" not in latent:
        raise TypeError("h3_latent must be a ComfyUI LATENT mapping")
    samples = latent["samples"]
    if getattr(samples, "is_nested", False):
        streams = samples.unbind()
        if len(streams) < 1:
            raise ValueError("H3 AV latent contains no video stream")
        samples = streams[0]
    if not isinstance(samples, torch.Tensor) or samples.ndim != 5:
        shape = getattr(samples, "shape", None)
        raise ValueError(
            "expected H3 video latent [B,24,T,H,W]; "
            f"received {type(samples).__name__} with shape {shape}"
        )
    if samples.shape[1] != 24:
        raise ValueError(
            f"expected 24 H3 video channels; received {samples.shape[1]}"
        )
    return samples


class H3ToLTXAdapterLoader:
    @classmethod
    def INPUT_TYPES(cls):
        models = _adapter_files()
        return {
            "required": {
                "model_name": (models or ["No model.safetensors found"],),
                "dtype": (["bfloat16", "float16", "float32"],),
            }
        }

    RETURN_TYPES = ("H3_LTX_ADAPTER",)
    RETURN_NAMES = ("adapter",)
    FUNCTION = "load"
    CATEGORY = "model/loaders"
    DESCRIPTION = (
        "Loads the released H3-to-LTX Conv3D latent adapter through "
        "ComfyUI model management."
    )

    def load(self, model_name: str, dtype: str):
        if model_name == "No model.safetensors found":
            raise FileNotFoundError(
                f"Place the adapter directory under {MODEL_ROOT}"
            )
        model_path = Path(
            folder_paths.get_full_path_or_raise(MODEL_FOLDER, model_name)
        )
        config_path = model_path.with_name("config.json")
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Expected config.json beside the checkpoint: {config_path}"
            )

        config = load_adapter_config(config_path)
        selected_dtype = DTYPES[dtype]
        offload_device = model_management.unet_offload_device()
        load_device = model_management.get_torch_device()

        model = build_model(config.model_config).eval().requires_grad_(False)
        model.to(device=offload_device, dtype=selected_dtype)
        state_dict = comfy.utils.load_torch_file(str(model_path), safe_load=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise ValueError(
                "adapter checkpoint does not match released architecture: "
                f"missing={missing}, unexpected={unexpected}"
            )
        del state_dict

        patcher = comfy.model_patcher.CoreModelPatcher(
            model,
            load_device=load_device,
            offload_device=offload_device,
        )
        loaded = LoadedH3ToLTXAdapter(
            model=model,
            patcher=patcher,
            config=config,
            dtype=selected_dtype,
            model_path=model_path,
        )
        LOGGER.info(
            "Loaded H3-to-LTX adapter metadata from %s (%s)",
            model_path,
            dtype,
        )
        return (loaded,)


class H3ToLTXLatentConvert:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "adapter": ("H3_LTX_ADAPTER",),
                "h3_latent": ("LATENT",),
                "input_normalization": (
                    ["normalized", "raw"],
                    {
                        "default": "normalized",
                        "tooltip": (
                            "ComfyUI H3 sampler outputs are normalized. "
                            "Use raw only for an explicitly unnormalized VAE latent."
                        ),
                    },
                ),
                "temporal_output": (
                    ["trim_to_source_duration", "keep_padded"],
                    {
                        "default": "trim_to_source_duration",
                        "tooltip": (
                            "124 H3 frames become 121 LTX frames when trimmed. "
                            "keep_padded retains the adapter's 129-frame grid."
                        ),
                    },
                ),
            },
            "optional": {
                "pixel_frames_override": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10000,
                        "step": 1,
                        "tooltip": "0 infers the H3 17k+5 duration from latent T.",
                    },
                )
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "INT")
    RETURN_NAMES = ("ltx_video_latent", "pixel_frames", "width", "height")
    FUNCTION = "convert"
    CATEGORY = "latent/conversion"
    DESCRIPTION = (
        "Converts normalized MiniMax H3 video latents directly into normalized "
        "LTX-2.5 video latents without an RGB decode/encode round trip."
    )

    def convert(
        self,
        adapter: LoadedH3ToLTXAdapter,
        h3_latent: dict,
        input_normalization: str,
        temporal_output: str,
        pixel_frames_override: int = 0,
    ):
        samples = _video_tensor(h3_latent)
        pixel_frames = (
            int(pixel_frames_override)
            if pixel_frames_override > 0
            else infer_h3_pixel_frames(int(samples.shape[2]))
        )
        pixel_height = int(samples.shape[3]) * 16
        pixel_width = int(samples.shape[4]) * 16
        if pixel_height % 32 or pixel_width % 32:
            raise ValueError(
                "adapter input dimensions must be divisible by 32 pixels; "
                f"received {pixel_width}x{pixel_height}"
            )

        target_frames = (pixel_frames - 1) // 8 + 1
        padded_frames = target_frames + int((pixel_frames - 1) % 8 != 0)
        target_height = pixel_height // 32
        target_width = pixel_width // 32
        bytes_per_value = torch.empty((), dtype=adapter.dtype).element_size()
        working_memory = (
            int(samples.shape[0])
            * 752
            * padded_frames
            * target_height
            * target_width
            * bytes_per_value
            * 8
        )
        model_management.load_models_gpu(
            [adapter.patcher],
            memory_required=working_memory,
            force_full_load=True,
        )

        output = convert_h3_to_ltx(
            adapter.model,
            samples,
            pixel_frames=pixel_frames,
            pixel_height=pixel_height,
            pixel_width=pixel_width,
            input_normalization=input_normalization,
            trim_to_source_duration=(
                temporal_output == "trim_to_source_duration"
            ),
            device=adapter.patcher.load_device,
            dtype=adapter.dtype,
        )
        output = output.to(model_management.intermediate_device())
        return (
            {"samples": output},
            pixel_frames,
            pixel_width,
            pixel_height,
        )


NODE_CLASS_MAPPINGS = {
    "H3ToLTXAdapterLoader": H3ToLTXAdapterLoader,
    "H3ToLTXLatentConvert": H3ToLTXLatentConvert,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ToLTXAdapterLoader": "Load H3-to-LTX Latent Adapter",
    "H3ToLTXLatentConvert": "H3-to-LTX Latent Convert",
}
