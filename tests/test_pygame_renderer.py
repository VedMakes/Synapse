from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from racing_event.config import SimulationConfig
from racing_event.evaluation import evaluate_policy
from racing_event.pygame_renderer import PygameReplayRenderer, run_pygame
from racing_event.replay import ReplayClock, sample_replay
from racing_event.track import TrackGeometry
from racing_event.trajectory import ReplayWriter, load_replay, replay_frame, replay_header, replay_result
from racing_event.vector_env import VectorizedRacingEnv


ROOT = Path(__file__).resolve().parents[1]
TRACK = TrackGeometry.from_file(ROOT / "tracks/train/synthetic_oval.json")
CONFIG = SimulationConfig.from_file(ROOT / "config/simulation.json")


def coast(observations):
    return [[0.0, 0.0] for _ in observations]


class PygameRendererTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "replay.json"
        header = replay_header(TRACK, CONFIG, participant_id="pygame-test", mode="evaluation")
        with ReplayWriter(self.path, header) as writer:
            result = evaluate_policy(VectorizedRacingEnv(TRACK, CONFIG, n_envs=1), coast, max_steps=2, frame_sink=writer.append)
            writer.finish(replay_result(result, CONFIG.tick_hz))
        self.data = load_replay(self.path)

    def test_headless_renderer_draws_without_mutating_replay(self) -> None:
        before = copy.deepcopy(self.data)
        pygame.init()
        try:
            surface = pygame.display.set_mode((640, 480))
            renderer = PygameReplayRenderer(self.data, surface)
            renderer.draw(sample_replay(self.data, ReplayClock(CONFIG.tick_hz, self.data["frames"][-1]["tick"])))
            self.assertEqual(self.data, before)
        finally:
            pygame.quit()

    def test_command_line_headless_smoke(self) -> None:
        run_pygame(self.path, size=(640, 480), headless=True)

    def test_renderer_uses_recorded_track_and_pose_only(self) -> None:
        pygame.init()
        try:
            surface = pygame.display.set_mode((640, 480))
            renderer = PygameReplayRenderer(self.data, surface)
            self.assertGreater(renderer._scale, 0.0)
            sampled = sample_replay(self.data, ReplayClock(CONFIG.tick_hz, self.data["frames"][-1]["tick"]))
            self.assertEqual(sampled["frame"]["tick"], 0)
            renderer.draw(sampled)
        finally:
            pygame.quit()


if __name__ == "__main__":
    unittest.main()
