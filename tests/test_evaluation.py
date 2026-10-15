from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from racing_event.evaluation import evaluate_policy, validate_evaluation_contract
from racing_event.env import RacingEnv
from racing_event.config import SimulationConfig
from racing_event.observation import RunningObservationNormalizer
from racing_event.track import TrackGeometry
from racing_event.vector_env import VectorizedRacingEnv


ROOT = Path(__file__).resolve().parents[1]
TRACK = TrackGeometry.from_file(ROOT / "tracks" / "train" / "synthetic_oval.json")
CONFIG = SimulationConfig.from_file(ROOT / "config" / "simulation.json")


class ZeroPolicy:
    def __call__(self, observations: np.ndarray) -> np.ndarray:
        return np.zeros((observations.shape[0], 2), dtype=np.float64)


class EvaluationTests(unittest.TestCase):
    def make_env(self, *, max_steps: int | None = None) -> VectorizedRacingEnv:
        config = CONFIG
        if max_steps is not None:
            config = replace(
                CONFIG,
                episode=replace(CONFIG.episode, max_steps=max_steps),
            )
        return VectorizedRacingEnv(TRACK, config, n_envs=1)

    def test_evaluation_is_deterministic_and_does_not_update_normalizer(self) -> None:
        env = self.make_env(max_steps=3)
        normalizer = RunningObservationNormalizer(env.observation_size)
        before = normalizer.state_dict()
        first = evaluate_policy(env, ZeroPolicy(), normalizer=normalizer)
        after_first = normalizer.state_dict()
        second = evaluate_policy(env, ZeroPolicy(), normalizer=normalizer)

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.status, "dnf_timeout")
        self.assertEqual(first.steps, 3)
        self.assertEqual(first.lap_time_s, None)
        self.assertEqual(before, after_first)
        self.assertEqual(len(first.trajectory_hash), 64)

    def test_contract_validation_rejects_changed_config_or_track(self) -> None:
        env = self.make_env()
        validate_evaluation_contract(env, env.training_contract)
        with self.assertRaises(ValueError):
            validate_evaluation_contract(
                env,
                replace(env.training_contract, config_hash="0" * 64),
            )
        with self.assertRaises(ValueError):
            validate_evaluation_contract(
                env,
                replace(env.training_contract, track_hashes=("0" * 64,)),
            )

    def test_evaluation_reset_rejects_noncanonical_spawn(self) -> None:
        scalar = RacingEnv(TRACK, CONFIG)
        with self.assertRaises(ValueError):
            scalar.reset(evaluation=True, initial_speed=1.0)
        with self.assertRaises(ValueError):
            self.make_env().reset(evaluation=True, lateral_offset=0.1)

    def test_stuck_outcome_is_distinct_from_timeout(self) -> None:
        env = self.make_env()
        result = evaluate_policy(env, ZeroPolicy(), max_steps=CONFIG.episode.stuck_steps + 1)
        self.assertEqual(result.status, "dnf_stuck")
        self.assertGreaterEqual(result.steps, CONFIG.episode.stuck_steps)


if __name__ == "__main__":
    unittest.main()
