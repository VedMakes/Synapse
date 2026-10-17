"""Read-only Pygame presentation for validated SYNAPSE replays.

Pygame consumes the Python-authored replay exchange.  It does not import an
environment, step physics, infer collisions, or calculate results.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping

from .replay import ReplayClock, sample_replay
from .trajectory import load_replay


class PygameReplayRenderer:
    """Draw one validated replay frame into a Pygame surface."""

    def __init__(self, data: Mapping[str, Any], surface: Any):
        import pygame

        self.data = data
        self.surface = surface
        self.width, self.height = surface.get_size()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 19)
        self.track = data["track"]
        self._origin_x, self._origin_y, self._scale = self._fit_track()
        self.path = [self._screen_point(float(frame["x"]), float(frame["y"])) for frame in data["frames"]]

    def _fit_track(self) -> tuple[float, float, float]:
        points: list[tuple[float, float]] = []
        for name in ("left_boundary", "right_boundary", "barrier_left", "barrier_right", "centerline"):
            points.extend((float(x), float(y)) for x, y in self.track.get(name, []))
        if not points:
            raise ValueError("replay track has no drawable geometry")
        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_y = max(point[1] for point in points)
        padding = 58.0
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)
        scale = min((self.width - 2.0 * padding) / span_x, (self.height - 2.0 * padding) / span_y)
        origin_x = padding - min_x * scale
        origin_y = self.height - padding + min_y * scale
        return origin_x, origin_y, scale

    def _screen_point(self, x: float, y: float) -> tuple[int, int]:
        return round(self._origin_x + x * self._scale), round(self._origin_y - y * self._scale)

    def _polyline(self, points: list[list[float]] | list[tuple[float, float]]) -> list[tuple[int, int]]:
        return [self._screen_point(float(x), float(y)) for x, y in points]

    def _draw_track(self, pygame: Any) -> None:
        colors = {
            "left_boundary": (62, 226, 210),
            "right_boundary": (62, 226, 210),
            "barrier_left": (255, 91, 132),
            "barrier_right": (255, 91, 132),
            "centerline": (76, 92, 112),
        }
        for name in ("barrier_left", "barrier_right", "left_boundary", "right_boundary", "centerline"):
            points = self._polyline(self.track.get(name, []))
            if len(points) >= 2:
                width = 4 if "barrier" in name else 2
                pygame.draw.lines(self.surface, colors[name], True, points, width)

    def _draw_path(self, pygame: Any) -> None:
        if len(self.path) >= 2:
            pygame.draw.lines(self.surface, (255, 209, 102), False, self.path, 2)

    def _draw_car(self, pygame: Any, pose: Mapping[str, float], off_track: bool) -> None:
        x = float(pose["x"])
        y = float(pose["y"])
        heading = float(pose["heading"])
        half_length = float(self.data["car"]["length_m"]) / 2.0
        half_width = float(self.data["car"]["width_m"]) / 2.0
        forward = (math.cos(heading), math.sin(heading))
        lateral = (-math.sin(heading), math.cos(heading))
        corners = []
        for longitudinal, transverse in (
            (half_length, half_width),
            (half_length, -half_width),
            (-half_length, -half_width),
            (-half_length, half_width),
        ):
            corners.append(self._screen_point(
                x + longitudinal * forward[0] + transverse * lateral[0],
                y + longitudinal * forward[1] + transverse * lateral[1],
            ))
        body_color = (255, 113, 148) if off_track else (69, 243, 255)
        pygame.draw.polygon(self.surface, (8, 14, 22), corners)
        pygame.draw.polygon(self.surface, body_color, corners, 3)
        nose = self._screen_point(x + half_length * forward[0], y + half_length * forward[1])
        pygame.draw.circle(self.surface, (255, 209, 102), nose, 4)

    def _text(self, pygame: Any, value: str, position: tuple[int, int], *, small: bool = False) -> None:
        font = self.small_font if small else self.font
        self.surface.blit(font.render(value, True, (232, 244, 255)), position)

    def draw(self, sampled: Mapping[str, Any]) -> None:
        import pygame

        self.surface.fill((5, 8, 14))
        self._draw_track(pygame)
        self._draw_path(pygame)
        frame = sampled["frame"]
        pose = sampled["pose"]
        self._draw_car(pygame, pose, bool(frame["off_track"]))
        result = self.data["result"]
        events = ", ".join(frame["events"]) or "none"
        self._text(pygame, f"SYNAPSE / {self.data['track']['meta']['display_name']}", (18, 14))
        self._text(pygame, f"tick {frame['tick']} / {result['steps']}   speed {pose['speed']:.2f} m/s", (18, 43), small=True)
        self._text(pygame, f"status {result['status']}   events {events}", (18, 66), small=True)
        self._text(pygame, f"trajectory {result['trajectory_hash'][:16]}…", (18, self.height - 28), small=True)


def run_pygame(
    replay_path: str | Path,
    *,
    size: tuple[int, int] = (1280, 800),
    speed: float = 1.0,
    headless: bool = False,
    start_seconds: float = 0.0,
    max_frames: int | None = None,
) -> None:
    """Run an interactive replay or a deterministic one-frame headless smoke."""

    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    data = load_replay(replay_path)
    pygame.init()
    try:
        surface = pygame.display.set_mode(size)
        renderer = PygameReplayRenderer(data, surface)
        clock = ReplayClock(data["tick_hz"], data["frames"][-1]["tick"], seconds=start_seconds, speed=speed, playing=not headless)
        if headless:
            renderer.draw(sample_replay(data, clock))
            pygame.display.flip()
            return
        frame_count = 0
        running = True
        wall_clock = pygame.time.Clock()
        while running:
            delta = wall_clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        clock.toggle()
                    elif event.key == pygame.K_r:
                        clock.restart()
                    elif event.key == pygame.K_LEFT:
                        clock.seek_seconds(clock.seconds - 1.0)
                    elif event.key == pygame.K_RIGHT:
                        clock.seek_seconds(clock.seconds + 1.0)
                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                        clock.set_speed(min(clock.speed * 2.0, 16.0))
                    elif event.key == pygame.K_MINUS:
                        clock.set_speed(max(clock.speed / 2.0, 0.125))
            clock.advance(delta)
            renderer.draw(sample_replay(data, clock))
            pygame.display.flip()
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break
    finally:
        pygame.quit()


__all__ = ["PygameReplayRenderer", "run_pygame"]
