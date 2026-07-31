import math
import tempfile
import unittest
from pathlib import Path

import torch

from atom.convert import (
    encode_weight_matrix,
    convert_state_dict,
    convert_checkpoint,
    save_conversion,
)


class TestEncodeWeightMatrix(unittest.TestCase):
    def test_sign_as_phase(self):
        w = torch.tensor([[1.0, -2.0], [0.5, -0.5]])
        amp, phase = encode_weight_matrix(w, phase_bits=None)
        self.assertTrue(torch.allclose(amp, w.abs()))
        self.assertTrue(torch.allclose(phase[0, 0], torch.tensor(0.0)))
        self.assertTrue(torch.allclose(phase[0, 1], torch.tensor(math.pi)))

    def test_default_8bit_quant(self):
        w = torch.randn(4, 4)
        amp, phase = encode_weight_matrix(w, phase_bits=8)
        # Only two ideal levels 0 and pi, quantised onto 256-level grid
        self.assertEqual(amp.shape, w.shape)
        self.assertTrue(torch.isfinite(phase).all())


class TestConvertStateDict(unittest.TestCase):
    def test_llama_style_names(self):
        state = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(32, 32),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(32, 32),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(32, 32),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(32, 32),
            "model.layers.0.mlp.up_proj.weight": torch.randn(64, 32),  # skip
        }
        result = convert_state_dict(state, phase_bits=8)
        self.assertEqual(result.meta["n_converted"], 4)
        self.assertIn(
            "model.layers.0.self_attn.q_proj.weight", result.weights
        )

    def test_gpt2_fused_c_attn(self):
        d = 24
        state = {
            "transformer.h.0.attn.c_attn.weight": torch.randn(3 * d, d),
            "transformer.h.0.attn.c_proj.weight": torch.randn(d, d),
        }
        result = convert_state_dict(state, phase_bits=8)
        # c_attn splits into q,k,v + c_proj as o
        self.assertGreaterEqual(result.meta["n_converted"], 3)

    def test_roundtrip_save_load_path(self):
        state = {
            "blocks.0.attn.q_proj.weight": torch.randn(16, 16),
            "blocks.0.attn.k_proj.weight": torch.randn(16, 16),
        }
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "model.pt"
            torch.save(state, ckpt)
            result = convert_checkpoint(ckpt, phase_bits=8)
            out = Path(tmp) / "optical"
            save_conversion(result, out)
            self.assertTrue((out / "optical_weights.pt").exists())
            self.assertTrue((out / "meta.json").exists())
            self.assertEqual(result.meta["n_converted"], 2)


if __name__ == "__main__":
    unittest.main()
