from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from racing_event.config import SimulationConfig
from racing_event.observation import RunningObservationNormalizer, TrainingContract, build_observation_schema


ROOT = Path(__file__).resolve().parents[1]
CONFIG = SimulationConfig.from_file(ROOT / "config" / "simulation.json")


class ObservationContractTests(unittest.TestCase):
    def test_schema_has_fixed_names_bounds_and_dtype(self) -> None:
        schema = build_observation_schema(CONFIG)
        self.assertEqual(schema.version, "1.0")
        self.assertEqual(schema.size, 23)
        self.assertEqual(schema.names[:4], ("speed", "longitudinal_velocity", "lateral_velocity", "yaw_rate"))
        self.assertEqual(schema.names[-1], "local_width")
        self.assertTrue(np.all(schema.lows <= schema.highs))

    def test_normalizer_and_contract_round_trip(self) -> None:
        schema = build_observation_schema(CONFIG)
        normalizer = RunningObservationNormalizer(schema.size)
        values = np.linspace(-0.5, 0.5, schema.size, dtype=np.float32)
        normalizer.update(np.stack([values, values * 0.5]))
        normalizer.freeze()
        contract = TrainingContract.for_config(CONFIG, track_ids=("oval",), track_hashes=("a" * 64,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            contract.save(path, normalizer=normalizer)
            loaded_contract, loaded_normalizer = TrainingContract.load(path)
        self.assertEqual(loaded_contract.to_dict()["action"]["shape"], [2])
        self.assertEqual(loaded_contract.observation.names, schema.names)
        self.assertIsNotNone(loaded_normalizer)
        assert loaded_normalizer is not None
        np.testing.assert_array_equal(normalizer.transform(values, training=False), loaded_normalizer.transform(values, training=False))


if __name__ == "__main__":
    unittest.main()
