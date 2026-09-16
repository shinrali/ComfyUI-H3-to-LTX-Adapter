import torch

from h3_ltx_adapter.model import build_model


def test_small_model_contract():
    model = build_model(
        {
            "model_type": "tiny",
            "in_channels": 384,
            "out_channels": 128,
            "width": 16,
            "num_blocks": 1,
            "expansion": 2,
            "groups": 16,
        }
    )
    output = model(torch.randn(1, 384, 2, 2, 2))
    assert output.shape == (1, 128, 2, 2, 2)
