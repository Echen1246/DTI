from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraIntrinsics:
    focal_length_px_x: float
    focal_length_px_y: float | None = None

    @property
    def focal_length_px(self) -> float:
        return self.focal_length_px_x if self.focal_length_px_y is None else (self.focal_length_px_x + self.focal_length_px_y) / 2


@dataclass(frozen=True)
class SizePrior:
    class_name: str
    width_m: float
    height_m: float | None = None


DEFAULT_DRONE_SIZE_PRIORS = {
    "small_quad": SizePrior("small_quad", width_m=0.35, height_m=0.12),
    "consumer_quad": SizePrior("consumer_quad", width_m=0.60, height_m=0.18),
    "large_quad": SizePrior("large_quad", width_m=1.20, height_m=0.35),
    "fixed_wing_small": SizePrior("fixed_wing_small", width_m=1.50, height_m=0.25),
}


def estimate_range_m(
    bbox_width_px: float,
    bbox_height_px: float,
    intrinsics: CameraIntrinsics,
    size_prior: SizePrior,
    use_height: bool = False,
) -> float | None:
    """Estimate distance from pinhole geometry: Z = f * real_size / observed_size."""
    if use_height and size_prior.height_m and bbox_height_px > 0:
        return intrinsics.focal_length_px * size_prior.height_m / bbox_height_px
    if bbox_width_px <= 0:
        return None
    return intrinsics.focal_length_px * size_prior.width_m / bbox_width_px


def estimate_range_interval_m(
    bbox_width_px: float,
    intrinsics: CameraIntrinsics,
    candidate_priors: list[SizePrior] | None = None,
) -> tuple[float, float] | None:
    priors = candidate_priors or list(DEFAULT_DRONE_SIZE_PRIORS.values())
    estimates = [
        estimate_range_m(bbox_width_px, bbox_width_px, intrinsics, prior)
        for prior in priors
    ]
    estimates = [value for value in estimates if value is not None]
    if not estimates:
        return None
    return min(estimates), max(estimates)
