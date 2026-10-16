from __future__ import annotations

import unittest

from tools.participant_ui import PAGE, build_training_command, parse_update_line, public_maps


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


if __name__ == "__main__":
    unittest.main()
