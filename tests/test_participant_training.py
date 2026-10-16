from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.build_training_manifest import build_manifest
from tools.run_reference_baseline import DEFAULT_CONFIG, DEFAULT_REWARD, DEFAULT_TRACK_DIR, run
from tools.train_participant import build_training_command


ROOT = Path(__file__).resolve().parents[1]


class ParticipantTrainingTests(unittest.TestCase):
    def test_public_training_manifest_has_only_valid_train_maps(self) -> None:
        manifest = build_manifest(DEFAULT_TRACK_DIR, "test-release")
        self.assertEqual(manifest["release_id"], "test-release")
        self.assertFalse(manifest["private_tracks_included"])
        self.assertEqual([item["split"] for item in manifest["maps"]], ["train"] * 3)
        self.assertTrue(all(len(item["canonical_content_hash"]) == 64 for item in manifest["maps"]))
        self.assertTrue(all(len(item["delivered_file_sha256"]) == 64 for item in manifest["maps"]))

    def test_reference_baseline_resumes_persistent_budget(self) -> None:
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

    def test_participant_launcher_forwards_owned_trainer(self) -> None:
        args = Namespace(
            trainer=ROOT / "examples" / "participant_train.py",
            reward=DEFAULT_REWARD,
            track_dir=DEFAULT_TRACK_DIR,
            config=DEFAULT_CONFIG,
            output_dir=ROOT / "participant_run",
            steps=8,
            budget_steps=16,
            n_envs=1,
            rollout_length=8,
            seed=17,
        )
        command = build_training_command(args)
        self.assertTrue(command[1].endswith("examples/participant_train.py"))
        self.assertIn("--steps", command)
        with self.assertRaises(ValueError):
            args.trainer = ROOT.parent / "outside.py"
            build_training_command(args)

    def test_participant_launcher_validates_inputs_and_allows_no_reward(self) -> None:
        args = Namespace(
            trainer=ROOT / "examples" / "participant_train.py",
            reward=None,
            track_dir=DEFAULT_TRACK_DIR,
            config=DEFAULT_CONFIG,
            output_dir=ROOT / "participant_run",
            steps=1,
            budget_steps=1,
            n_envs=1,
            rollout_length=1,
            seed=17,
        )
        command = build_training_command(args)
        self.assertNotIn("--reward", command)
        args.config = ROOT / "artifacts" / "missing-config.json"
        with self.assertRaisesRegex(ValueError, "config does not exist"):
            build_training_command(args)

    def test_reference_checkpoint_recovers_after_policy_publish_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "recover"
            args = Namespace(
                reward=DEFAULT_REWARD,
                track_dir=DEFAULT_TRACK_DIR,
                tracks=None,
                config=DEFAULT_CONFIG,
                output_dir=output_dir,
                budget_steps=8,
                steps=4,
                n_envs=1,
                rollout_length=4,
                seed=17,
            )
            from tools import run_reference_baseline

            original = run_reference_baseline.FixedMLPActorCritic.save_artifact
            calls = 0

            def flaky_save(policy, *save_args, **save_kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected checkpoint interruption")
                return original(policy, *save_args, **save_kwargs)

            with patch.object(run_reference_baseline.FixedMLPActorCritic, "save_artifact", flaky_save):
                with self.assertRaisesRegex(OSError, "injected"):
                    run_reference_baseline.run(args)
            self.assertTrue((output_dir / "training_state.json").exists())
            resumed = run_reference_baseline.run(args)
            self.assertEqual(resumed["budget_used"], 8)
            self.assertEqual(len((output_dir / "training_metrics.jsonl").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
