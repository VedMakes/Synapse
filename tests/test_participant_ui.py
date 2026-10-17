from __future__ import annotations

import unittest
import os
import json
from pathlib import Path
import sys
import tempfile
import time

from tools.participant_ui import PAGE, ROOT, TrainingProcess, build_training_command, parse_update_line, public_maps


class ParticipantUiTests(unittest.TestCase):
    def test_update_line_is_parsed_for_dashboard_metrics(self) -> None:
        line = "update 3: steps 768/10000 return=-2.500 lap=0.125 speed=21.250 m/s crash=0.250 off_track=0.375"
        self.assertEqual(
            parse_update_line(line),
            {
                "update": 3,
                "budget_used": 768,
                "budget_steps": 10000,
                "return": -2.5,
                "lap": 0.125,
                "speed": 21.25,
                "crash": 0.25,
                "off_track": 0.375,
            },
        )

    def test_command_is_workspace_bound_and_uses_existing_cli(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output_dir = Path(directory).resolve()
            command = build_training_command({"output_dir": str(output_dir), "trainer": "examples/participant_train.py", "steps": 16})
        self.assertTrue(command[1].endswith("tools/train_participant.py"))
        self.assertIn("--trainer", command)
        self.assertTrue(command[command.index("--trainer") + 1].endswith("/examples/participant_train.py"))
        self.assertIn("--output-dir", command)
        self.assertEqual(Path(command[command.index("--output-dir") + 1]), output_dir)
        with self.assertRaises(ValueError):
            build_training_command({"output_dir": "../../outside"})

    def test_page_contains_participant_contract_and_controls(self) -> None:
        for text in (
            "Start / resume participant training",
            "Cumulative budget",
            "Reward function · signal weights",
            "Forward-facing ray sensors",
            "Watch reference artifact for map",
            "Set up / repair Python",
            "/api/status",
            'id="ray-diagram"',
            'role="img"',
            "renderRayDiagram",
            'id="ray-summary"',
            "The cyan line is 0° forward",
            "curvature_matched_speed",
            "Measures abrupt steering changes; larger means rougher",
            '["ray-count", "ray-angles"].forEach(id => $(id).addEventListener("input", renderRayDiagram))',
            'viewBox="0 0 640 430"',
            'id="ray-callouts"',
            "ray-label-box",
            "ray-endpoint",
            "car-wheel",
            "Positive angles sweep left",
        ):
            self.assertIn(text, PAGE)

    def test_ui_writes_weighted_reward_and_ray_config_snapshots(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output_dir = Path(directory).resolve()
            command = build_training_command({
                "output_dir": str(output_dir),
                "trainer": "examples/participant_train.py",
                "reward_weights": {"progress": "4.5", "crash": "-20"},
                "ray_count": "3",
                "ray_angles_deg": "-30,0,30",
                "steps": 1,
            })
            reward_path = Path(command[command.index("--reward") + 1])
            config_path = Path(command[command.index("--config") + 1])
            self.assertEqual(json.loads(reward_path.read_text())["weights"]["progress"], 4.5)
            self.assertEqual(json.loads(config_path.read_text())["sensors"]["ray_angles_deg"], [-30.0, 0.0, 30.0])
            self.assertTrue((output_dir / "run_identity.json").exists())

    def test_ui_preserves_non_sorted_ray_order_in_identity(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output_dir = Path(directory).resolve()
            build_training_command({
                "output_dir": str(output_dir),
                "trainer": "examples/participant_train.py",
                "ray_count": 3,
                "ray_angles_deg": "30,-30,0",
                "steps": 1,
            })
            config = json.loads((output_dir / "run_config.json").read_text())
            identity = json.loads((output_dir / "run_identity.json").read_text())
        self.assertEqual(config["sensors"]["ray_angles_deg"], [30.0, -30.0, 0.0])
        self.assertEqual(identity["ray_angles_deg"], [30.0, -30.0, 0.0])
        self.assertEqual(identity["ray_count"], 3)

    def test_ui_rejects_ray_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            with self.assertRaisesRegex(ValueError, "ray_count"):
                build_training_command({
                    "output_dir": directory,
                    "trainer": "examples/participant_train.py",
                    "ray_count": 2,
                    "ray_angles_deg": "-30,0,30",
                })
            with self.assertRaisesRegex(ValueError, "ray_count"):
                build_training_command({
                    "output_dir": directory,
                    "trainer": "examples/participant_train.py",
                    "ray_count": 3.5,
                    "ray_angles_deg": "-30,0,30",
                })

    def test_public_map_selector_has_three_development_maps(self) -> None:
        maps = public_maps()
        self.assertEqual(len(maps), 3)
        self.assertEqual({item["id"] for item in maps}, {"synthetic_oval", "synthetic_flowing", "synthetic_tight_loop"})

    def test_numeric_inputs_reject_fractional_boolean_and_infinite_values(self) -> None:
        for value in (1.5, True, float("nan"), float("inf"), "not-a-number"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    build_training_command({"steps": value})

    def test_missing_paths_are_rejected_before_start(self) -> None:
        with self.assertRaisesRegex(ValueError, "trainer does not exist"):
            build_training_command({"trainer": "artifacts/does-not-exist.py"})
        with self.assertRaisesRegex(ValueError, "config does not exist"):
            build_training_command({"config": "artifacts/does-not-exist.json"})

    def test_stop_reaches_a_terminal_stopped_state(self) -> None:
        process = TrainingProcess()
        process.start_command(
            [sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(10)"],
            kind="test",
        )
        for _ in range(20):
            if process.snapshot()["running"]:
                break
            time.sleep(0.01)
        self.assertTrue(process.snapshot()["running"])
        process.stop()
        for _ in range(100):
            snapshot = process.snapshot()
            if not snapshot["running"]:
                break
            time.sleep(0.02)
        self.assertFalse(snapshot["running"])
        self.assertEqual(snapshot["state"], "stopped")

    def test_stop_terminates_the_trainer_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pid_path = Path(directory) / "child.pid"
            code = (
                "import pathlib, subprocess, sys, time; "
                "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)']); "
                "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(10)"
            )
            process = TrainingProcess()
            process.start_command([sys.executable, "-c", code, str(pid_path)], kind="tree-test")
            for _ in range(100):
                if pid_path.exists():
                    break
                time.sleep(0.02)
            self.assertTrue(pid_path.exists())
            child_pid = int(pid_path.read_text())
            process.stop()
            for _ in range(150):
                try:
                    os.kill(child_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.02)
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)


if __name__ == "__main__":
    unittest.main()
