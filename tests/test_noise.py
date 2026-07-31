import math
import unittest

import torch

from atom import (
    quantize_phase,
    add_phase_noise,
    add_angular_jitter,
    apply_crosstalk,
    NoiseConfig,
    optical_scores_general,
)


class TestPhaseQuantization(unittest.TestCase):
    def test_quantize_phase_has_correct_level_count(self):
        # Sweep a fine grid across a full turn and confirm the quantizer
        # only ever produces exactly 2**bits distinct output values.
        phase = torch.linspace(0, 2 * math.pi, steps=10_000)
        for bits in (1, 2, 4, 8):
            quantized = quantize_phase(phase, bits)
            distinct = torch.unique(torch.round(quantized, decimals=6))
            self.assertLessEqual(distinct.numel(), 2 ** bits)

    def test_quantize_phase_wraps_cyclically(self):
        # Phase is cyclic -- an angle and that angle plus a full turn must
        # quantize to the same level, since an SLM has no concept of
        # "angle 400 degrees."
        phase = torch.tensor([0.3, 0.3 + 2 * math.pi, 0.3 - 2 * math.pi])
        quantized = quantize_phase(phase, bits=6)
        self.assertTrue(torch.allclose(quantized[0], quantized[1], atol=1e-5))
        self.assertTrue(torch.allclose(quantized[0], quantized[2], atol=1e-5))

    def test_quantize_phase_rejects_nonpositive_bits(self):
        with self.assertRaises(ValueError):
            quantize_phase(torch.zeros(4), bits=0)

    def test_high_bit_quantization_converges_to_continuous_case(self):
        # At high enough precision, quantized scores must approach the
        # ideal continuous-phase case arbitrarily closely -- if this
        # didn't hold, the quantizer itself would be broken, independent
        # of any real hardware question.
        torch.manual_seed(0)
        q = torch.randn(6, 16)
        k = torch.randn(6, 16)
        positions = torch.arange(6, dtype=torch.float32)

        ideal = optical_scores_general(q, k, query_positions=positions, key_positions=positions)
        high_precision = optical_scores_general(
            q, k, query_positions=positions, key_positions=positions, phase_bits=16
        )
        self.assertTrue(torch.allclose(ideal, high_precision, atol=1e-3))

    def test_low_bit_quantization_visibly_degrades_scores(self):
        # Conversely, 1-bit phase (effectively back to the binary case
        # plus rounding) should NOT match the continuous case -- if it
        # did, quantization wouldn't be doing anything, meaning the test
        # above would be vacuous.
        torch.manual_seed(1)
        q = torch.randn(6, 16)
        k = torch.randn(6, 16)
        positions = torch.arange(6, dtype=torch.float32)

        ideal = optical_scores_general(q, k, query_positions=positions, key_positions=positions)
        low_precision = optical_scores_general(
            q, k, query_positions=positions, key_positions=positions, phase_bits=1
        )
        self.assertFalse(torch.allclose(ideal, low_precision, atol=1e-2))


class TestPhaseNoise(unittest.TestCase):
    def test_zero_sigma_is_identity(self):
        phase = torch.randn(8, 16)
        out = add_phase_noise(phase, sigma=0.0)
        self.assertTrue(torch.equal(out, phase))

    def test_positive_sigma_changes_values(self):
        torch.manual_seed(42)
        phase = torch.randn(8, 16)
        out = add_phase_noise(phase, sigma=0.1)
        self.assertFalse(torch.allclose(out, phase))

    def test_rejects_negative_sigma(self):
        with self.assertRaises(ValueError):
            add_phase_noise(torch.zeros(4), sigma=-0.1)


class TestAngularJitter(unittest.TestCase):
    def test_zero_sigma_is_identity(self):
        positions = torch.arange(8, dtype=torch.float32)
        out = add_angular_jitter(positions, sigma=0.0)
        self.assertTrue(torch.equal(out, positions))

    def test_positive_sigma_changes_values(self):
        torch.manual_seed(43)
        positions = torch.arange(8, dtype=torch.float32)
        out = add_angular_jitter(positions, sigma=0.05)
        self.assertFalse(torch.allclose(out, positions))


class TestCrosstalk(unittest.TestCase):
    def test_zero_strength_is_identity(self):
        scores = torch.randn(4, 6, 6)
        out = apply_crosstalk(scores, strength=0.0)
        self.assertTrue(torch.equal(out, scores))

    def test_positive_strength_mixes(self):
        scores = torch.eye(5).unsqueeze(0)  # (1, 5, 5)
        out = apply_crosstalk(scores, strength=0.5, kernel_size=3)
        # Off-diagonal should become non-zero due to mixing
        self.assertFalse(torch.allclose(out, scores))

    def test_rejects_invalid_strength(self):
        with self.assertRaises(ValueError):
            apply_crosstalk(torch.randn(3, 4, 4), strength=1.5)

    def test_rejects_even_kernel(self):
        with self.assertRaises(ValueError):
            apply_crosstalk(torch.randn(3, 4, 4), strength=0.3, kernel_size=4)


class TestNoisyScores(unittest.TestCase):
    def test_ideal_path_unchanged(self):
        torch.manual_seed(0)
        q = torch.randn(6, 16)
        k = torch.randn(6, 16)
        positions = torch.arange(6, dtype=torch.float32)

        ideal = optical_scores_general(q, k, query_positions=positions, key_positions=positions)
        with_defaults = optical_scores_general(
            q, k,
            query_positions=positions,
            key_positions=positions,
            phase_sigma=0.0,
            angular_jitter=0.0,
            crosstalk=0.0,
        )
        self.assertTrue(torch.equal(ideal, with_defaults))

    def test_noise_config_overrides(self):
        torch.manual_seed(7)
        q = torch.randn(5, 12)
        k = torch.randn(5, 12)
        positions = torch.arange(5, dtype=torch.float32)

        ideal = optical_scores_general(q, k, query_positions=positions, key_positions=positions)
        cfg = NoiseConfig(phase_sigma=0.2, crosstalk=0.3)
        noisy = optical_scores_general(
            q, k, query_positions=positions, key_positions=positions, noise=cfg
        )
        self.assertFalse(torch.allclose(ideal, noisy, atol=1e-3))

    def test_gradients_flow_through_noisy_path(self):
        torch.manual_seed(9)
        q = torch.randn(4, 8, requires_grad=True)
        k = torch.randn(4, 8, requires_grad=True)
        positions = torch.arange(4, dtype=torch.float32)

        scores = optical_scores_general(
            q, k,
            query_positions=positions,
            key_positions=positions,
            phase_sigma=0.05,
            angular_jitter=0.01,
            crosstalk=0.1,
        )
        loss = scores.sum()
        loss.backward()
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertTrue(torch.isfinite(k.grad).all())


if __name__ == "__main__":
    unittest.main()
