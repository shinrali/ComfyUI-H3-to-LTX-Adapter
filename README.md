# ComfyUI H3-to-LTX Adapter

ComfyUI custom nodes for the public
[`Efficient-Large-Model/H3-to-LTX-Latent-Adapter`](https://huggingface.co/Efficient-Large-Model/H3-to-LTX-Latent-Adapter).

The adapter translates a sampled MiniMax H3 video latent directly into the
normalized LTX-2.5 Conv VAE latent space. It avoids decoding H3 to RGB and
encoding those frames again with the LTX VAE.

## Status

Released. The implementation follows the NVIDIA Sol-H3-Spark geometry and
Conv3D architecture and has completed an end-to-end run in a PC ComfyUI
installation. The adapter conversion preserved the source content in that
test. Identity changes introduced later by an LTX refinement sampler are a
property of that downstream workflow and are outside this node package's
conversion boundary.

## Nodes

- **Load H3-to-LTX Latent Adapter** loads the released `config.json` and
  `model.safetensors` through ComfyUI model management.
- **H3-to-LTX Latent Convert** accepts either a plain H3 video latent or the
  joint H3 AV latent and outputs a plain normalized LTX video latent.

The converter does not translate H3 audio. Separate the AV latent with the
stock **Separate AV Latent** node when the H3 audio must be decoded or retained
for final muxing.

## Install

Clone the repository into the ComfyUI custom node directory:

```powershell
cd C:\path\to\ComfyUI\custom_nodes
git clone https://github.com/shinrali/ComfyUI-H3-to-LTX-Adapter.git
```

The resulting directory should be:

```text
ComfyUI/custom_nodes/ComfyUI-H3-to-LTX-Adapter/
```

Install the small Python dependency with the Python used by ComfyUI:

```powershell
cd C:\path\to\ComfyUI\custom_nodes\ComfyUI-H3-to-LTX-Adapter
C:\path\to\python.exe -m pip install -r requirements.txt
```

Restart ComfyUI after installation.

## Model

Download both `config.json` and `model.safetensors` from:

<https://huggingface.co/Efficient-Large-Model/H3-to-LTX-Latent-Adapter>

Place them together:

```text
ComfyUI/models/h3_ltx_adapters/H3-to-LTX-Latent-Adapter/
├── config.json
└── model.safetensors
```

Model weights are intentionally excluded from this repository.

## Minimal graph

```text
MiniMax H3 Sampler output (joint AV latent)
    │
    ├── Separate AV Latent ── audio ── H3 audio decode / retain for mux
    │                    └── video
    │                         ▼
    │              MiniMax H3 Latent Upscaler 3D (x2)
    │                         │
Load H3-to-LTX Adapter ───────┤
                              ▼
                   H3-to-LTX Latent Convert
                              │ normalized LTX video latent
                              │
                              ▼
                   LTX-2.5 three-step refiner
                              │
                              ▼
                   LTX-2.5 Conv VAE decode
```

Recommended initial converter settings:

```text
input_normalization: normalized
temporal_output: trim_to_source_duration
pixel_frames_override: 0
```

For a standard H3 `124`-frame latent:

```text
H3 draft input:          [B, 24, 37, 24, 42]   (672x384)
H3 x2 upscaled latent:   [B, 24, 37, 48, 84]   (1344x768)
Adapter native output:   [B, 128, 17, 24, 42]
Trimmed LTX output:      [B, 128, 16, 24, 42]
LTX pixel output:        1344x768, 121 frames
```

The validated Sol-H3-Spark order is **H3 learned latent upscale first, then
H3-to-LTX conversion**. Do not replace it with adapter conversion followed by
`LTXVLatentUpsampler`: that produces the same final tensor dimensions through a
different, unvalidated feature path and can weaken identity and scene retention.
The H3 upscaler node and checkpoint are separate dependencies and are not
bundled with this adapter package.

The adapter pads the temporal grid upward before conversion. The default
`trim_to_source_duration` removes the final padded LTX latent slot so a
124-frame H3 draft becomes the normal 121-frame LTX output. Use `keep_padded`
only when deliberately testing the 129-frame padded output.

## LTX second-stage starting point

The public Sol-H3-Spark recipe uses a content-agnostic quality prompt because
the converted latent already carries scene and motion information:

```text
4K, refined, high quality, cinematic detail, clean textures, natural motion.
```

Recommended initial refinement schedule:

```text
CFG: 1.0
sigmas: 0.909375, 0.725, 0.421875, 0.0
updates: 3
sampler: euler
```

The validated refiner uses the LTX-2.5 dev transformer with
`ltx-2.5-22b-distilled-lora-450-bf16.safetensors` at strength `0.8`. A full
distilled transformer or `euler_cfg_pp` is not the same recipe.

Use the normal LTX text encoder and connector for the first ComfyUI test. The
offline fixed-prompt context cache from the standalone Sol-H3-Spark runtime is
not a standard ComfyUI `CONDITIONING` object and is not loaded by these nodes.

## Important boundaries

- Input must be the H3 **video** latent with 24 channels. H3 audio is not an
  adapter input.
- ComfyUI H3 sampler output should use `normalized`. `raw` exists only for an
  explicitly unnormalized H3 VAE latent.
- Adapter output is already normalized for LTX. Do not normalize it again.
- Pixel width and height must be divisible by 32.
- The node infers the H3 `17k+5` pixel duration from its `5k+2` latent grid.
- This package does not install or patch Sol-Attn, H3, LTX, or ComfyUI itself.
- This package converts video latents only. It does not translate H3 text,
  image-reference, video-reference, or audio-reference conditioning into LTX
  conditioning.
- Sol-H3-Spark passes generated H3 latents, not H3 reference-conditioning
  activations, into Stage 2. Any identity behavior after conversion depends on
  the downstream LTX model, conditioning, sampler, and sigma schedule.

## Development checks

Inside a Python environment containing PyTorch:

```bash
python -m pip install pytest safetensors
python -m pytest -q
python -m compileall -q .
```

## Provenance and licensing

The architecture, latent statistics, geometry, and source behavior are adapted
from the Apache-2.0 licensed
[`NVlabs/Sana`](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/Sol-H3-Spark)
Sol-H3-Spark implementation. See `NOTICE`.

The adapter model weights, MiniMax H3, LTX-2.5, and ComfyUI retain their own
licenses and model-use terms. No third-party model weights are redistributed.
