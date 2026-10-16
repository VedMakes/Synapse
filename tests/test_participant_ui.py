from __future__ import annotations

import unittest
import os
from pathlib import Path
import sys
import tempfile
import time

from tools.participant_ui import PAGE, TrainingProcess, build_training_command, parse_update_line, public_maps


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
        command = build_training_command({"output_dir": "ui-run", "trainer": "examples/participant_train.py", "steps": 16})
        self.assertTrue(command[1].endswith("tools/train_participant.py"))
        self.assertIn("--trainer", command)
        self.assertTrue(command[command.index("--trainer") + 1].endswith("/examples/participant_train.py"))
        self.assertIn("--output-dir", command)
        self.assertTrue(command[command.index("--output-dir") + 1].endswith("/ui-run"))
        with self.assertRaises(ValueError):
            build_training_command({"output_dir": "../../outside"})

    def test_page_contains_participant_contract_and_controls(self) -> None:
        for text in ("Start / resume participant training", "Cumulative budget", "Your model and reward", "Watch reference artifact for map", "Set up / repair Python", "/api/status"):
            self.assertIn(text, PAGE)

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
