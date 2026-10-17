from __future__ import annotations

import math
import unittest

from racing_event.replay import ReplayClock, sample_replay


def replay_data() -> dict[str, object]:
    def frame(tick: int, x: float, heading: float) -> dict[str, object]:
        return {
            "tick": tick,
            "x": x,
            "y": 0.0,
            "heading": heading,
            "speed": 2.0,
            "lateral_velocity": 0.0,
            "yaw_rate": 0.0,
            "steer": 0.0,
            "steering_command": 0.0,
            "throttle_brake": 0.5,
            "off_track": False,
            "events": [],
            "terminated": False,
            "truncated": False,
        }

    return {"tick_hz": 10, "frames": [frame(0, 0.0, math.radians(179)), frame(1, 10.0, math.radians(-179)), frame(2, 20.0, math.radians(-170))]}


class ReplayClockTests(unittest.TestCase):
    def test_exact_ticks_and_final_frame_clamp(self) -> None:
        data = replay_data()
        clock = ReplayClock(10, 2)
        self.assertEqual(sample_replay(data, clock)["frame_index"], 0)
        clock.seek_tick(1)
        self.assertEqual(sample_replay(data, clock)["frame_index"], 1)
        clock.seek_seconds(99)
        sampled = sample_replay(data, clock)
        self.assertEqual(sampled["frame_index"], 2)
        self.assertEqual(sampled["alpha"], 0.0)

    def test_shortest_heading_interpolation_and_playback_speed(self) -> None:
        data = replay_data()
        clock = ReplayClock(10, 2, seconds=0.05)
        sampled = sample_replay(data, clock)
        self.assertGreater(abs(sampled["pose"]["heading"]), math.radians(178))
        clock.playing = True
        clock.set_speed(2.0)
        clock.advance(0.1)
        self.assertAlmostEqual(clock.seconds, 0.2)

    def test_invalid_clock_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ReplayClock(0, 1)
        clock = ReplayClock(10, 2)
        with self.assertRaises(ValueError):
            clock.set_speed(0)
        with self.assertRaises(ValueError):
            clock.seek_seconds(float("nan"))


if __name__ == "__main__":
    unittest.main()
