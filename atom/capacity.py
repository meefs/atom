"""M#-aware capacity estimates for holographic weight storage.

The geometric ceiling (90T per cm3 under the simulation parameters) ignores
the dynamic-range limit of real photorefractive media. The number of equal-
efficiency holograms that can be multiplexed is set by the material M#:

    eta ≈ (M# / M)**2

so the usable channel count is M ≈ M# / sqrt(eta_min), where eta_min is the
lowest diffraction efficiency still readable at acceptable SNR.

Defaults are conservative values typical of iron-doped lithium niobate
(Fe:LiNbO3) in the 90-degree geometry. All parameters are exposed so the
numbers can be updated when better measured M# or eta_min values are
available.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CapacityParams:
    """Geometry and material parameters for a capacity estimate.

    Defaults match the simulation geometry already used in the rest of the
    project and conservative Fe:LiNbO3 figures from the holographic-storage
    literature.
    """

    # Geometry (same numbers as docs/benchmarks.md)
    side_cm: float = 1.0
    pixel_um: float = 1.0
    layer_um: float = 10.0
    angular_range_deg: float = 90.0
    angular_step_deg: float = 0.1

    # Material (Fe:LiNbO3, 90-degree geometry, conservative)
    m_number: float = 2.0
    eta_min: float = 1e-4

    def geometric_channels(self) -> int:
        return int(self.angular_range_deg / self.angular_step_deg)

    def spatial_pixels_per_layer(self) -> int:
        side_um = self.side_cm * 1e4
        n = int(side_um / self.pixel_um)
        return n * n

    def depth_layers(self) -> int:
        return int((self.side_cm * 1e4) / self.layer_um)

    def volume_cm3(self) -> float:
        return self.side_cm ** 3


def max_usable_channels(m_number: float, eta_min: float) -> float:
    """Maximum number of equal-efficiency holograms from the M# relation.

    M ≈ M# / sqrt(eta_min). Returns a float so callers can decide how to
    round; the value is already an upper bound under the equal-efficiency
    assumption.
    """
    if m_number <= 0:
        raise ValueError("m_number must be positive")
    if eta_min <= 0 or eta_min > 1:
        raise ValueError("eta_min must be in (0, 1]")
    return m_number / math.sqrt(eta_min)


def geometric_capacity(params: CapacityParams | None = None) -> float:
    """Pure geometric ceiling (no dynamic-range limit)."""
    p = params or CapacityParams()
    return float(
        p.depth_layers() * p.geometric_channels() * p.spatial_pixels_per_layer()
    )


def usable_capacity(params: CapacityParams | None = None) -> float:
    """Dynamic-range limited capacity under the M# model.

    Angular channels are limited by max_usable_channels; spatial and depth
    dimensions stay geometric. The result is the number of independent
    weight values that can be stored at the chosen minimum diffraction
    efficiency.
    """
    p = params or CapacityParams()
    channels = min(p.geometric_channels(), max_usable_channels(p.m_number, p.eta_min))
    return float(p.depth_layers() * channels * p.spatial_pixels_per_layer())


def capacity_summary(params: CapacityParams | None = None) -> dict[str, float]:
    """Convenience report: geometric vs usable and the ratio."""
    p = params or CapacityParams()
    geo = geometric_capacity(p)
    use = usable_capacity(p)
    return {
        "volume_cm3": p.volume_cm3(),
        "m_number": p.m_number,
        "eta_min": p.eta_min,
        "geometric_capacity": geo,
        "usable_capacity": use,
        "usable_fraction": use / geo if geo > 0 else 0.0,
        "max_usable_channels": max_usable_channels(p.m_number, p.eta_min),
        "geometric_channels": float(p.geometric_channels()),
    }
