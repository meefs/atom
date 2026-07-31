"""Convert attention weights to optical phase encodings.

Without --model: runs on a small synthetic Llama-style state dict (CI-safe).
With --model /path/to/checkpoint: loads your local weights (safetensors or .pt).

Default phase_bits=8 matches the quantisation sweep knee.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from atom.convert import convert_checkpoint, convert_state_dict, save_conversion


def synthetic_state() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    d = 64
    return {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(d, d),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(d, d),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(d, d),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(d, d),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(2 * d, d),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to local checkpoint file or directory (safetensors / .pt)",
    )
    p.add_argument("--out", type=str, default="optical_weights_out")
    p.add_argument("--phase-bits", type=int, default=8)
    args = p.parse_args()

    if args.model:
        print(f"Loading checkpoint from {args.model}")
        result = convert_checkpoint(args.model, phase_bits=args.phase_bits)
    else:
        print("No --model given; using synthetic Llama-style weights")
        result = convert_state_dict(synthetic_state(), phase_bits=args.phase_bits)

    print(f"Converted {result.meta['n_converted']} tensors")
    print(f"Skipped   {result.meta['n_skipped']} non-attention keys")
    print(f"phase_bits={result.meta['phase_bits']}")
    for name in list(result.meta["converted_keys"])[:12]:
        ow = result.weights[name]
        print(f"  {name}  shape={ow.shape}")
    if result.meta["n_converted"] > 12:
        print("  ...")

    out = Path(args.out)
    save_conversion(result, out)
    print(f"Wrote {out / 'optical_weights.pt'} and meta.json")


if __name__ == "__main__":
    main()
