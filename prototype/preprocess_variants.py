"""Throwaway prototype for #23: torch semantics candidates for `preprocess.onnx`.

NOT production code. Three variants share one strip geometry (pole, radii, 1 deg/column):

- `CleanPreprocess(center="ideal")`: one-shot sampling of minimap and asset at the strip
  grid; alpha compositing in the strip domain, one rounding per output. The reference
  center is the exact (x, y) and the scale is exact.
- `CleanPreprocess(center="cv2")`: same structure, but the asset sample positions are the
  analytic composition of the current cv2 chain (integer center `round(x)`,
  width `round(118*scale)`, half-pixel INTER_LINEAR resize), so only the intermediate
  images disappear and the sample positions are unchanged.
- `ReplicaPreprocess`: the current cv2 order, two sampling stages with uint8 rounding in
  between (black composite -> crop/resize -> observed backdrop -> unwrap).

Geometry conventions match `endfield.polar.unwrap` / `endfield.ref`: coordinates are
OpenCV pixel-center coordinates, `GridSample(align_corners=0)` with grid
`(2*p + 1) / size - 1`. Asset out-of-bounds reads as zero (reference gap), which is what
the cv2 zero-padded crop does; `padding_mode="zeros"` on the raw asset reproduces that
for every position the ring actually samples.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from endfield.polar import IMG_H, IMG_W, INNER_R, OUTER_R
from endfield.ref import ROI_H, ROI_POLE, ROI_W


def strip_roi_uv() -> torch.Tensor:
    """[IMG_H, IMG_W, 2] strip coordinates in ROI pixel-center space (u right, v down)."""
    step = (OUTER_R - INNER_R) / IMG_H
    radii = INNER_R + step * (np.arange(IMG_H, dtype=np.float32) + 0.5)
    theta = np.deg2rad(np.arange(IMG_W, dtype=np.float32))
    u = ROI_POLE[0] + radii[:, None] * np.sin(theta)[None, :]
    v = ROI_POLE[1] - radii[:, None] * np.cos(theta)[None, :]
    return torch.from_numpy(np.stack([u, v], axis=-1))


def _normalized(u: torch.Tensor, v: torch.Tensor, width, height) -> torch.Tensor:
    """Pixel-center coordinates -> GridSample align_corners=0 grid [1,H,W,2]."""
    return torch.stack([(2.0 * u + 1.0) / width - 1.0, (2.0 * v + 1.0) / height - 1.0], dim=-1)[
        None
    ]


def _to_uint8(value: torch.Tensor) -> torch.Tensor:
    return torch.round(value).clamp(0.0, 255.0).to(torch.uint8)


class CleanPreprocess(nn.Module):
    """One-shot sampling + strip-domain compositing; no intermediate images."""

    def __init__(self, center: str = "ideal") -> None:
        super().__init__()
        if center not in ("ideal", "cv2"):
            raise ValueError(f"unknown center mode {center!r}")
        self.center = center
        self.register_buffer("roi_uv", strip_roi_uv(), persistent=False)

    def forward(
        self,
        minimap: torch.Tensor,
        asset: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u, v = self.roi_uv[..., 0], self.roi_uv[..., 1]
        obs = F.grid_sample(
            minimap.permute(0, 3, 1, 2).float(),
            _normalized(u, v, ROI_W, ROI_H),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).permute(0, 2, 3, 1)

        if self.center == "ideal":
            au = x + (u - ROI_POLE[0]) * scale
            av = y + (v - ROI_POLE[1]) * scale
        else:
            width = torch.round(ROI_W * scale)
            height = torch.round(ROI_H * scale)
            au = torch.round(x) - torch.floor(width / 2.0) + (u + 0.5) * (width / ROI_W) - 0.5
            av = torch.round(y) - torch.floor(height / 2.0) + (v + 0.5) * (height / ROI_H) - 0.5

        grid = _normalized(au, av, asset.shape[2], asset.shape[1])
        planes = asset.permute(0, 3, 1, 2).float()
        rgb = F.grid_sample(
            planes[:, :3], grid, mode="bilinear", padding_mode="zeros", align_corners=False
        ).permute(0, 2, 3, 1)
        alpha = F.grid_sample(
            planes[:, 3:4], grid, mode="bilinear", padding_mode="zeros", align_corners=False
        ).permute(0, 2, 3, 1)

        alpha_fraction = alpha / 255.0
        composed = rgb * alpha_fraction + obs * (1.0 - alpha_fraction)
        return _to_uint8(obs), torch.cat([_to_uint8(composed), _to_uint8(alpha)], dim=-1)


class ReplicaPreprocess(nn.Module):
    """Current cv2 order, replicated point by point as an ONNX-exportable graph."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("roi_uv", strip_roi_uv(), persistent=False)
        self.register_buffer("idx_u", torch.arange(ROI_W, dtype=torch.float32), persistent=False)
        self.register_buffer("idx_v", torch.arange(ROI_H, dtype=torch.float32), persistent=False)

    def forward(
        self,
        minimap: torch.Tensor,
        asset: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # composite_on_black on the full asset, rounded to uint8 (current order)
        black = _to_uint8(asset[..., :3].float() * (asset[..., 3:4].float() / 255.0))
        alpha_plane = asset[..., 3:4]

        # crop_centered + cv2.resize(INTER_LINEAR, half-pixel) to the ROI grid
        width = torch.round(ROI_W * scale)
        height = torch.round(ROI_H * scale)
        x0 = torch.round(x) - torch.floor(width / 2.0)
        y0 = torch.round(y) - torch.floor(height / 2.0)
        su = x0 + (self.idx_u + 0.5) * (width / ROI_W) - 0.5
        sv = y0 + (self.idx_v + 0.5) * (height / ROI_H) - 0.5
        gx = (2.0 * su + 1.0) / asset.shape[2] - 1.0
        gy = (2.0 * sv + 1.0) / asset.shape[1] - 1.0
        grid1 = torch.stack(
            [gx[None, :].expand(ROI_H, ROI_W), gy[:, None].expand(ROI_H, ROI_W)], dim=-1
        )[None]
        black_roi = _to_uint8(
            F.grid_sample(
                black.permute(0, 3, 1, 2).float(),
                grid1,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )
        alpha_roi = _to_uint8(
            F.grid_sample(
                alpha_plane.permute(0, 3, 1, 2).float(),
                grid1,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )

        # compose_observed_backdrop, then polar unwrap (cv2 BORDER_REPLICATE)
        obs_roi = minimap.permute(0, 3, 1, 2)
        composed = _to_uint8(
            black_roi.float() + obs_roi.float() * (1.0 - alpha_roi.float() / 255.0)
        )
        grid2 = _normalized(self.roi_uv[..., 0], self.roi_uv[..., 1], ROI_W, ROI_H)
        observed = _to_uint8(
            F.grid_sample(
                obs_roi.float(), grid2, mode="bilinear", padding_mode="border", align_corners=False
            )
        ).permute(0, 2, 3, 1)
        ref_bgr = _to_uint8(
            F.grid_sample(
                composed.float(),
                grid2,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
        ).permute(0, 2, 3, 1)
        ref_alpha = _to_uint8(
            F.grid_sample(
                alpha_roi.float(),
                grid2,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
        ).permute(0, 2, 3, 1)
        return observed, torch.cat([ref_bgr, ref_alpha], dim=-1)


VARIANTS = {
    "clean_ideal": lambda: CleanPreprocess("ideal"),
    "clean_cv2align": lambda: CleanPreprocess("cv2"),
    "replica": ReplicaPreprocess,
}
