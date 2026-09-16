import torch

from h3_ltx_adapter.adapter import expected_shapes
from h3_ltx_adapter.geometry import (
    align_h3_to_ltx,
    h3_temporal_positions,
    infer_h3_pixel_frames,
    target_ltx_latent_frames,
)


def test_h3_124_frame_grid():
    assert len(h3_temporal_positions(124)) == 37
    assert infer_h3_pixel_frames(37) == 124
    assert target_ltx_latent_frames(124) == 16


def test_expected_960x544_shapes():
    shapes = expected_shapes(124, 544, 960)
    assert shapes["h3"] == [24, 37, 34, 60]
    assert shapes["aligned_h3"] == [384, 17, 17, 30]
    assert shapes["ltx"] == [128, 17, 17, 30]


def test_alignment_shape():
    source = torch.randn(1, 24, 37, 4, 6)
    aligned = align_h3_to_ltx(
        source,
        pixel_frames=124,
        target_height=2,
        target_width=3,
    )
    assert aligned.shape == (1, 384, 17, 2, 3)
