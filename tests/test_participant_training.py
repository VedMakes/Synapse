from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

from tools.build_training_manifest import build_manifest
from tools.train_participant import DEFAULT_CONFIG, DEFAULT_REWARD, DEFAULT_TRACK_DIR, run


ROOT = Path(__file__).resolve().parents[1]


class ParticipantTrainingTests(unittest.TestCase):
    def test_public_training_manifest_has_only_valid_train_maps(self) -> None:
        manifest = build_manifest(DEFAULT_TRACK_DIR, "test-release")
        self.assertEqual(manifest["release_id"], "test-release")
        self.assertFalse(manifest["private_tracks_included"])
        self.assertEqual([item["split"] for item in manifest["maps"]], ["train"] * 3)
        self.assertTrue(all(len(item["canonical_content_hash"]) == 64 for item in manifest["maps"]))
        self.assertTrue(all(len(item["delivered_file_sha256"]) == 64 for item in manifest["maps"]))

    def test_participant_runner_resumes_persistent_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            args = Namespace(
                reward=DEFAULT_REWARD,
                track_dir=DEFAULT_TRACK_DIR,
                tracks=None,
                config=DEFAULT_CONFIG,
                output_dir=output_dir,
                budget_steps=16,
                steps=8,
                n_envs=1,
                rollout_length=8,
                seed=17,
            )
            first = run(args)
            second = run(args)
            self.assertEqual(first["budget_used"], 8)
            self.assertEqual(second["budget_used"], 16)
            self.assertTrue((output_dir / "policy.pt").exists())
            self.assertTrue((output_dir / "training_contract.json").exists())
            self.assertEqual(len((output_dir / "training_metrics.jsonl").read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
