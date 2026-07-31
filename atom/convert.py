"""Convert digital attention weights into optical phase encodings.

Loads a local checkpoint (PyTorch state dict or safetensors directory) and
maps attention projection matrices into the amplitude/phase representation
used by the optical score path. Default phase quantisation is 8 bits, matching
the knee in examples/05_phase_quantization_sweep.py.

This module does not download models. Point it at weights already on disk.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .noise import quantize_phase

# Common attention projection name patterns (HF / GPT-2 / Llama-style)
# Note: fused GPT-2 c_attn is handled separately and must not match q_proj.
_Q_PATTERNS = re.compile(
    r"(?:^|[.])(?:q_proj|query|q_lin|to_q)$", re.IGNORECASE
)
_K_PATTERNS = re.compile(
    r"(?:^|[.])(?:k_proj|key|k_lin|to_k)$", re.IGNORECASE
)
_V_PATTERNS = re.compile(
    r"(?:^|[.])(?:v_proj|value|v_lin|to_v)$", re.IGNORECASE
)
_O_PATTERNS = re.compile(
    r"(?:^|[.])(?:o_proj|out_proj|c_proj|to_out\.0)$", re.IGNORECASE
)


@dataclass
class OpticalWeightTensor:
    """One weight matrix encoded for the optical path."""

    name: str
    amplitude: torch.Tensor
    phase: torch.Tensor
    shape: tuple
    phase_bits: int | None

    def to_complex(self) -> torch.Tensor:
        return self.amplitude * torch.exp(1j * self.phase)


@dataclass
class ConversionResult:
    weights: dict
    meta: dict


def encode_weight_matrix(
    weight: torch.Tensor,
    phase_bits: int | None = 8,
) -> tuple:
    """Map a real weight matrix to amplitude + phase.

    Sign is stored as binary phase (0 / pi). If phase_bits is set, phase is
    quantised to that many levels around the circle (default 8).
    """
    w = weight.detach().float().cpu()
    amplitude = w.abs()
    phase = torch.where(w >= 0, torch.zeros_like(w), torch.full_like(w, math.pi))
    if phase_bits is not None:
        phase = quantize_phase(phase, phase_bits)
    return amplitude, phase


def _torch_load(path: str):
    """Load a pickle checkpoint; weights_only when supported."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_state_dict(path: Path) -> dict:
    path = Path(path)
    if path.is_file():
        if path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError as e:
                raise ImportError(
                    "safetensors is required to load .safetensors files"
                ) from e
            return load_file(str(path))
        obj = _torch_load(str(path))
        if isinstance(obj, dict) and "state_dict" in obj:
            return obj["state_dict"]
        if isinstance(obj, dict):
            return {k: v for k, v in obj.items() if torch.is_tensor(v)}
        raise ValueError(f"Unsupported checkpoint format: {path}")

    st = list(path.glob("*.safetensors"))
    if st:
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError(
                "safetensors is required to load .safetensors files"
            ) from e
        out = {}
        for f in st:
            out.update(load_file(str(f)))
        return out

    bins = list(path.glob("*.bin")) + list(path.glob("pytorch_model*.bin"))
    if bins:
        out = {}
        for f in bins:
            part = _torch_load(str(f))
            if isinstance(part, dict):
                out.update({k: v for k, v in part.items() if torch.is_tensor(v)})
        return out

    raise FileNotFoundError(f"No weights found under {path}")


def _classify_key(key: str) -> str | None:
    k = key
    if k.endswith(".weight"):
        k = k[: -len(".weight")]
    # Fused GPT-2 style must be checked before generic q/k/v patterns
    if k.endswith("c_attn") or k.endswith(".attn.c_attn"):
        return "c_attn"
    if _O_PATTERNS.search(k):
        return "o"
    if _Q_PATTERNS.search(k):
        return "q"
    if _K_PATTERNS.search(k):
        return "k"
    if _V_PATTERNS.search(k):
        return "v"
    return None


def convert_state_dict(
    state: dict,
    phase_bits: int | None = 8,
    include_output_proj: bool = True,
) -> ConversionResult:
    """Convert attention-related tensors in a state dict to optical form."""
    optical = {}
    skipped = []

    for key, tensor in state.items():
        if not torch.is_tensor(tensor) or tensor.ndim < 1:
            continue
        kind = _classify_key(key)
        if kind is None:
            skipped.append(key)
            continue
        if kind == "o" and not include_output_proj:
            skipped.append(key)
            continue

        if kind == "c_attn":
            if tensor.ndim != 2:
                skipped.append(key)
                continue
            d = tensor.shape[0] // 3
            if d * 3 != tensor.shape[0]:
                skipped.append(key)
                continue
            for name, chunk in zip(
                ("q", "k", "v"),
                (tensor[:d], tensor[d : 2 * d], tensor[2 * d :]),
            ):
                amp, phase = encode_weight_matrix(chunk, phase_bits=phase_bits)
                full_name = f"{key}.{name}"
                optical[full_name] = OpticalWeightTensor(
                    name=full_name,
                    amplitude=amp,
                    phase=phase,
                    shape=tuple(chunk.shape),
                    phase_bits=phase_bits,
                )
            continue

        amp, phase = encode_weight_matrix(tensor, phase_bits=phase_bits)
        optical[key] = OpticalWeightTensor(
            name=key,
            amplitude=amp,
            phase=phase,
            shape=tuple(tensor.shape),
            phase_bits=phase_bits,
        )

    meta = {
        "phase_bits": phase_bits,
        "n_converted": len(optical),
        "n_skipped": len(skipped),
        "converted_keys": sorted(optical.keys()),
    }
    return ConversionResult(weights=optical, meta=meta)


def convert_checkpoint(
    path,
    phase_bits: int | None = 8,
    include_output_proj: bool = True,
) -> ConversionResult:
    """Load a local checkpoint and convert attention weights."""
    state = _load_state_dict(Path(path))
    return convert_state_dict(
        state, phase_bits=phase_bits, include_output_proj=include_output_proj
    )


def save_conversion(result: ConversionResult, out_dir) -> None:
    """Write amplitude/phase tensors and metadata to a directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {}
    for name, ow in result.weights.items():
        safe = name.replace("/", "__").replace(".", "__")
        payload[f"{safe}.amplitude"] = ow.amplitude
        payload[f"{safe}.phase"] = ow.phase
    torch.save(payload, out / "optical_weights.pt")
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(result.meta, f, indent=2)
