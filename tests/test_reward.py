from __future__ import annotations

import math
import unittest

from racing_event.reward import RewardConfig


class RewardConfigTests(unittest.TestCase):
    def test_missing_signals_default_to_zero(self) -> None:
        config = RewardConfig(weights={"progress": 2})
        self.assertEqual(config.weights["progress"], 2.0)
        self.assertEqual(config.weights["speed"], 0.0)

    def test_non_finite_weights_are_rejected(self) -> None:
        for value in (math.nan, math.inf, -math.inf, "not-a-number"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    RewardConfig(weights={"progress": value})


if __name__ == "__main__":
    unittest.main()
