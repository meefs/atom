"""Noise modelling: physical constraints the pure-math simulator ignores.

Starts with phase quantization -- CONTRIBUTING.md flags this as the most
tractable open noise problem, and the natural next question after proving
the continuous-phase math works in the ideal case (see attention.py):
a real crystal can't write an infinite-precision phase angle. It writes
theta to some finite number of bits. This module answers "how many bits
before that actually matters."

Extended with:
- Additive Gaussian phase noise (write / SLM temporal instability)
- Angular (Bragg) position jitter
- Soft inter-channel crosstalk on attention scores
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def quantize_phase(phase: torch.Tensor, bits: int) -> torch.Tensor:
    """Quantize a phase angle (radians, any range) to `bits`-bit precision.

    Models a real spatial light modulator or crystal write mechanism that
    can only address 2**bits distinct phase levels around the unit circle,
    rather than an idealized continuous angle. Phase is wrapped to
    [0, 2*pi) before quantizing, since phase is cyclic -- an SLM has no
    concept of "phase 400 degrees," it only has 2**bits discrete steps
    around one full turn.
    """
    if bits <= 0:
        raise ValueError("bits must be positive")
    levels = 2 ** bits
    wrapped = torch.remainder(phase, 2 * math.pi)
    step = 2 * math.pi / levels
    quantized = torch.round(wrapped / step) * step
    # Rounding near the top of the circle can land exactly on 2*pi, which
    # is the same physical angle as 0 but a distinct float value -- wrap
    # again so that boundary collapses to the single level it actually is.
    return torch.remainder(quantized, 2 * math.pi)


def add_phase_noise(phase: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add independent Gaussian phase jitter (radians).

    Models write-precision noise, SLM temporal instability, or residual
    phase error after calibration. sigma is the standard deviation in
    radians. sigma=0 is a no-op.
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if sigma == 0:
        return phase
    noise = torch.randn_like(phase) * sigma
    return phase + noise


def add_angular_jitter(positions: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add Gaussian jitter to angular / Bragg positions.

    Models thermal drift or mechanical instability of the incidence angle.
    sigma is in the same units as `positions` (radians or arbitrary angle
    units used by the caller). sigma=0 is a no-op.
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if sigma == 0:
        return positions
    return positions + torch.randn_like(positions) * sigma


def apply_crosstalk(
    scores: torch.Tensor,
    strength: float,
    kernel_size: int = 3,
) -> torch.Tensor:
    """Apply soft leakage between neighbouring angular channels.

    `scores` has shape (..., query_seq, key_seq). Crosstalk mixes along
    the key dimension (the angular-multiplexed axis). `strength` in [0, 1]
    controls how much of the local neighbourhood leaks in
    (0 = pure, 1 = fully averaged over the kernel).

    Uses a simple box / uniform kernel for now; can be replaced with a
    measured Bragg selectivity curve later.
    """
    if strength < 0 or strength > 1:
        raise ValueError("strength must be in [0, 1]")
    if strength == 0 or kernel_size <= 1:
        return scores
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    pad = kernel_size // 2
    # scores: (..., Q, K)
    padded = torch.nn.functional.pad(scores, (pad, pad), mode="replicate")
    mixed = torch.zeros_like(scores)
    for i in range(kernel_size):
        mixed = mixed + padded[..., i : i + scores.shape[-1]]
    mixed = mixed / kernel_size

    return (1.0 - strength) * scores + strength * mixed


@dataclass
class NoiseConfig:
    """Bundle of optional noise parameters.

    All fields default to "ideal / off". Pass this (or individual kwargs)
    into the optical score functions.
    """

    phase_bits: int | None = None
    phase_sigma: float = 0.0
    angular_jitter: float = 0.0
    crosstalk: float = 0.0
    crosstalk_kernel: int = 3
