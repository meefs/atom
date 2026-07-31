import math
import unittest

from atom.capacity import (
    CapacityParams,
    max_usable_channels,
    geometric_capacity,
    usable_capacity,
    capacity_summary,
)


class TestMaxUsableChannels(unittest.TestCase):
    def test_basic_relation(self):
        # M = M# / sqrt(eta)
        self.assertAlmostEqual(max_usable_channels(2.0, 1e-4), 200.0)
        self.assertAlmostEqual(max_usable_channels(5.0, 1e-4), 500.0)

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            max_usable_channels(0.0, 1e-4)
        with self.assertRaises(ValueError):
            max_usable_channels(2.0, 0.0)
        with self.assertRaises(ValueError):
            max_usable_channels(2.0, 1.5)


class TestGeometricCapacity(unittest.TestCase):
    def test_default_matches_benchmarks(self):
        # Same arithmetic as docs/benchmarks.md: 1000 * 900 * 1e8 = 9e13
        self.assertEqual(geometric_capacity(), 9e13)

    def test_scales_with_volume(self):
        p1 = CapacityParams(side_cm=1.0)
        p2 = CapacityParams(side_cm=2.0)
        # Linear dimensions double -> volume *8, layers *2, pixels *4 -> *8
        self.assertAlmostEqual(geometric_capacity(p2) / geometric_capacity(p1), 8.0)


class TestUsableCapacity(unittest.TestCase):
    def test_limited_by_m_number(self):
        p = CapacityParams(m_number=2.0, eta_min=1e-4)
        # channels limited to 200 instead of 900
        expected = 1000 * 200 * 1e8
        self.assertAlmostEqual(usable_capacity(p), expected)

    def test_recovers_geometric_when_m_number_large(self):
        p = CapacityParams(m_number=1e6, eta_min=1e-4)
        self.assertAlmostEqual(usable_capacity(p), geometric_capacity(p))

    def test_summary_keys(self):
        s = capacity_summary()
        self.assertIn("geometric_capacity", s)
        self.assertIn("usable_capacity", s)
        self.assertIn("usable_fraction", s)
        self.assertLess(s["usable_fraction"], 1.0)
        self.assertGreater(s["usable_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
