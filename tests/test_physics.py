from __future__ import annotations

import unittest

import numpy as np

from racing_event.config import CarConfig
from racing_event.physics import CarState, step_car


class PhysicsTests(unittest.TestCase):
    def test_throttle_accelerates_and_brake_reduces_speed(self) -> None:
        config = CarConfig()
        state = CarState(0.0, 0.0, 0.0)
        for _ in range(20):
            state = step_car(state, np.asarray([0.0, 1.0]), config, 1.0 / 30.0)
        self.assertGreater(state.speed, 0.0)
        before_braking = state.speed
        for _ in range(10):
            state = step_car(state, np.asarray([0.0, -1.0]), config, 1.0 / 30.0)
        self.assertLess(state.speed, before_braking)

    def test_steering_rate_is_limited(self) -> None:
        config = CarConfig()
        state = CarState(0.0, 0.0, 0.0)
        next_state = step_car(state, np.asarray([1.0, 0.0]), config, 1.0 / 30.0)
        self.assertLessEqual(abs(next_state.steering_angle), config.steering_rate_rad_s / 30.0 + 1.0e-12)

    def test_same_state_and_action_are_bitwise_repeatable(self) -> None:
        config = CarConfig()
        action = np.asarray([0.25, 0.7])
        left = step_car(CarState(1.0, 2.0, 0.3, speed=4.0), action, config, 1.0 / 30.0)
        right = step_car(CarState(1.0, 2.0, 0.3, speed=4.0), action, config, 1.0 / 30.0)
        self.assertEqual(left.__dict__, right.__dict__)


if __name__ == "__main__":
    unittest.main()
