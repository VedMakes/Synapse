from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from racing_event.observation import TrainingContract
from tools.train_scripted_imitation import train


ROOT = Path(__file__).resolve().parents[1]


class ScriptedImitationTests(unittest.TestCase):
    def test_lightweight_imitation_run_writes_loadable_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "imitation"
            report = train(type("Args", (), {
                "track_dir": ROOT / "tracks/train",
                "config": ROOT / "config/simulation.json",
                "output_dir": output,
                "epochs": 2,
                "samples_per_track": 8,
                "seed": 17,
            })())
            self.assertEqual(report["samples"], 24)
            self.assertTrue((output / "policy.pt").exists())
            self.assertTrue((output / "policy.json").exists())
            contract, normalizer = TrainingContract.load(output / "training_contract.json")
        self.assertIsNone(normalizer)
        self.assertEqual(contract.track_ids, ("synthetic_flowing", "synthetic_oval", "synthetic_tight_loop"))


if __name__ == "__main__":
    unittest.main()
