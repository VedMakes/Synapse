"""Read-only Pygame presentation for validated SYNAPSE replays.

Pygame consumes the Python-authored replay exchange. It never steps physics,
infers contacts, or calculates results. Only position and heading interpolate.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
import math
import os
from pathlib import Path
from typing import Any, Mapping

from .replay import ReplayClock, sample_replay
from .trajectory import load_replay


MIN_WINDOW_SIZE = (640, 480)
MIN_SPEED = 0.125
MAX_SPEED = 16.0
BACKGROUND = (5, 8, 14)
PANEL = (15, 24, 36)
TEXT = (232, 244, 255)
MUTED = (158, 180, 202)
ACCENT = (69, 243, 255)
WARNING = (255, 113, 148)
GOLD = (255, 209, 102)
EVENT_LABELS = {
    "off_track_start": "Off track",
    "off_track_end": "Back on track",
    "barrier_contact": "Barrier contact",
    "lap_complete": "Lap complete",
}
RESULT_LABELS = {
    "finished": "FINISHED",
    "dnf_crash": "DNF - barrier contact",
    "dnf_stuck": "DNF - stuck",
    "dnf_backward": "DNF - backward",
    "dnf_timeout": "DNF - timeout",
}


class PygameReplayRenderer:
    """Present and navigate one recording; all seeks pause at a recorded tick."""

    def __init__(self, data: Mapping[str, Any], surface: Any, clock: ReplayClock | None = None):
        import pygame

        self.data = data
        self.track = data["track"]
        self.clock = clock if clock is not None else ReplayClock(data["tick_hz"], data["frames"][-1]["tick"])
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 19)
        # Scan bounded recording data once, never during animation or a seek.
        self.event_ticks = tuple(frame["tick"] for frame in data["frames"] if frame["events"])
        self._bounds = self._world_bounds()
        self.show_help = False
        self.dragging = False
        self.hover_event: int | None = None
        self.resize(surface)

    def _world_bounds(self) -> tuple[float, float, float, float]:
        points = [point for name in ("left_boundary", "right_boundary", "barrier_left", "barrier_right", "centerline")
                  for point in self.track.get(name, [])]
        if not points:
            raise ValueError("replay track has no drawable geometry")
        min_x = min(float(point[0]) for point in points)
        max_x = max(float(point[0]) for point in points)
        min_y = min(float(point[1]) for point in points)
        max_y = max(float(point[1]) for point in points)
        # Include the whole recorded path and a conservative footprint radius.
        # This also contains every intermediate shortest-angle pose, without
        # inflating the actual car drawn on screen.
        radius = math.hypot(self.data["car"]["length_m"], self.data["car"]["width_m"]) / 2
        for frame in self.data["frames"]:
            min_x, max_x = min(min_x, frame["x"] - radius), max(max_x, frame["x"] + radius)
            min_y, max_y = min(min_y, frame["y"] - radius), max(max_y, frame["y"] + radius)
        return min_x, max_x, min_y, max_y

    def resize(self, surface: Any) -> None:
        """Rebuild the static layer only when the surface size changes."""
        import pygame

        self.surface = surface
        size = surface.get_size()
        if getattr(self, "_cached_size", None) == size:
            return
        self.width, self.height = size
        if self.width < MIN_WINDOW_SIZE[0] or self.height < MIN_WINDOW_SIZE[1]:
            raise ValueError("viewer size must be at least 640x480")
        self._cached_size = size
        self.header_rect = pygame.Rect(0, 0, self.width, 80)
        self.footer_rect = pygame.Rect(0, self.height - 166, self.width, 166)
        panel_width = 216 if self.width < 960 else 272
        content_height = self.footer_rect.top - self.header_rect.bottom - 24
        self.panel_rect = pygame.Rect(self.width - panel_width - 12, 92, panel_width, content_height)
        self.track_rect = pygame.Rect(12, 92, self.panel_rect.left - 24, content_height)
        self.scrubber_rect = pygame.Rect(20, self.footer_rect.top + 27, self.width - 40, 32)
        self.buttons: dict[str, Any] = {}
        x = 16
        for action, width in (("play", 78), ("restart", 76), ("previous_event", 94), ("next_event", 94),
                              ("slower", 36), ("speed", 58), ("faster", 36), ("help", 70)):
            self.buttons[action] = pygame.Rect(x, self.footer_rect.top + 86, width, 34)
            x += width + 8
        self._origin_x, self._origin_y, self._scale = self._fit_track()
        self.path = [self._screen_point(float(frame["x"]), float(frame["y"])) for frame in self.data["frames"]]
        self._background = pygame.Surface(size)
        self._background.fill(BACKGROUND)
        self._background.set_clip(self.track_rect)
        self._draw_track(pygame, self._background)
        if len(self.path) >= 2:
            pygame.draw.lines(self._background, GOLD, False, self.path, 1)
        self._background.set_clip(None)
        # Symbols preserve event type without relying on color. Multiple events
        # on one tick share a seek target and remain listed in the readout.
        self.marker_rects = {
            tick: pygame.Rect(self._tick_x(tick) - 6, self.scrubber_rect.top + 16, 13, 16)
            for tick in self.event_ticks
        }
        self._draw_timeline_background(pygame)
        self.dragging = False
        self.hover_event = None

    def _fit_track(self) -> tuple[float, float, float]:
        min_x, max_x, min_y, max_y = self._bounds
        area = self.track_rect.inflate(-28, -28)
        scale = min(area.width / max(max_x - min_x, 1.0), area.height / max(max_y - min_y, 1.0))
        return (area.centerx - (min_x + max_x) * scale / 2,
                area.centery + (min_y + max_y) * scale / 2, scale)

    def _screen_point(self, x: float, y: float) -> tuple[int, int]:
        return round(self._origin_x + x * self._scale), round(self._origin_y - y * self._scale)

    def _polyline(self, points: Any) -> list[tuple[int, int]]:
        return [self._screen_point(float(x), float(y)) for x, y in points]

    def _draw_track(self, pygame: Any, target: Any) -> None:
        colors = {"left_boundary": (62, 226, 210), "right_boundary": (62, 226, 210),
                  "barrier_left": (255, 91, 132), "barrier_right": (255, 91, 132), "centerline": (76, 92, 112)}
        for name in ("barrier_left", "barrier_right", "left_boundary", "right_boundary", "centerline"):
            points = self._polyline(self.track.get(name, []))
            if len(points) >= 2:
                pygame.draw.lines(target, colors[name], True, points, 2 if "barrier" in name else 1)

    def _tick_x(self, tick: float) -> int:
        return round(self.scrubber_rect.left + tick / self.clock.final_tick * (self.scrubber_rect.width - 1))

    def _draw_timeline_background(self, pygame: Any) -> None:
        y = self.scrubber_rect.top + 8
        pygame.draw.line(self._background, MUTED, (self.scrubber_rect.left, y), (self.scrubber_rect.right - 1, y), 3)
        for tick in self.event_ticks:
            x = self._tick_x(tick)
            events = self.data["frames"][tick]["events"]
            y = self.scrubber_rect.top + 23
            if "barrier_contact" in events:
                pygame.draw.line(self._background, WARNING, (x - 4, y - 4), (x + 4, y + 4), 2)
                pygame.draw.line(self._background, WARNING, (x + 4, y - 4), (x - 4, y + 4), 2)
            elif "lap_complete" in events:
                pygame.draw.polygon(self._background, GOLD, [(x, y - 5), (x + 5, y), (x, y + 5), (x - 5, y)])
            elif "off_track_start" in events:
                pygame.draw.polygon(self._background, WARNING, [(x - 4, y - 4), (x + 4, y - 4), (x, y + 4)])
            else:
                pygame.draw.polygon(self._background, ACCENT, [(x - 4, y + 4), (x + 4, y + 4), (x, y - 4)])

    def _draw_car(self, pygame: Any, pose: Mapping[str, float], off_track: bool) -> None:
        x, y, heading = float(pose["x"]), float(pose["y"]), float(pose["heading"])
        half_length = float(self.data["car"]["length_m"]) / 2
        half_width = float(self.data["car"]["width_m"]) / 2
        forward = (math.cos(heading), math.sin(heading))
        lateral = (-math.sin(heading), math.cos(heading))

        def point(longitudinal: float, transverse: float) -> tuple[int, int]:
            return self._screen_point(x + longitudinal * forward[0] + transverse * lateral[0],
                                      y + longitudinal * forward[1] + transverse * lateral[1])

        corners = [point(half_length, half_width), point(half_length, -half_width),
                   point(-half_length, -half_width), point(-half_length, half_width)]
        # No minimum-sized sprite, wheels, shadow or nose outside the recorded
        # metric rectangle: the displayed footprint remains geometry truth.
        pygame.draw.polygon(self.surface, WARNING if off_track else ACCENT, corners)
        pygame.draw.line(self.surface, BACKGROUND, point(half_length * 0.35, half_width * 0.65),
                         point(half_length * 0.35, -half_width * 0.65), 1)

    def _text(self, value: str, rect: Any, *, small: bool = True, color: Any = TEXT, centered: bool = False) -> None:
        font = self.small_font if small else self.font
        # Untrusted participant/track names can be long; fit them to their own
        # region rather than letting text escape over the track or controls.
        if font.size(value)[0] > rect.width:
            while value and font.size(value + "...")[0] > rect.width:
                value = value[:-1]
            value += "..."
        image = font.render(value, True, color)
        destination = image.get_rect(center=rect.center) if centered else image.get_rect(topleft=rect.topleft)
        self.surface.blit(image, destination)

    def _seek(self, tick: int) -> None:
        self.clock.pause()
        self.clock.seek_tick(tick)

    def _event_target(self, direction: int) -> int | None:
        position = self.clock.tick_position
        index = bisect_left(self.event_ticks, position) - 1 if direction < 0 else bisect_right(self.event_ticks, position)
        return self.event_ticks[index] if 0 <= index < len(self.event_ticks) else None

    def _enabled(self, action: str) -> bool:
        if action in ("previous_event", "next_event"):
            return self._event_target(-1 if action == "previous_event" else 1) is not None
        if action == "slower":
            return self.clock.speed > MIN_SPEED
        if action == "faster":
            return self.clock.speed < MAX_SPEED
        return action != "speed"

    def _activate(self, action: str) -> None:
        if not self._enabled(action):
            return
        if action == "play":
            if self.clock.tick_position >= self.clock.final_tick:
                self.clock.restart()
            else:
                self.clock.toggle()
        elif action == "restart":
            self.clock.restart()
        elif action in ("previous_event", "next_event"):
            target = self._event_target(-1 if action == "previous_event" else 1)
            if target is not None:
                self._seek(target)
        elif action in ("slower", "faster"):
            self.clock.set_speed(max(MIN_SPEED, min(MAX_SPEED, self.clock.speed * (0.5 if action == "slower" else 2))))
        elif action == "help":
            self.show_help = not self.show_help

    def _marker_at(self, position: tuple[int, int]) -> int | None:
        # Nearest marker wins when dense recordings put several on one pixel;
        # previous/next event still reaches every recorded event tick.
        candidates = (tick for tick, rect in self.marker_rects.items() if rect.collidepoint(position))
        return min(candidates, key=lambda tick: abs(self._tick_x(tick) - position[0]), default=None)

    def _scrub(self, x: int) -> None:
        fraction = (x - self.scrubber_rect.left) / (self.scrubber_rect.width - 1)
        self._seek(round(max(0.0, min(1.0, fraction)) * self.clock.final_tick))

    def handle_event(self, event: Any) -> bool:
        """Consume real Pygame input. Return False when the viewer should close."""
        import pygame

        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.VIDEORESIZE:
            size = (max(MIN_WINDOW_SIZE[0], event.w), max(MIN_WINDOW_SIZE[1], event.h))
            self.resize(pygame.display.set_mode(size, pygame.RESIZABLE))
        elif event.type == pygame.KEYDOWN:
            key = event.key
            if key in (pygame.K_ESCAPE, pygame.K_q):
                return False
            actions = {pygame.K_SPACE: "play", pygame.K_r: "restart", pygame.K_p: "previous_event", pygame.K_n: "next_event",
                       pygame.K_LEFTBRACKET: "slower", pygame.K_MINUS: "slower", pygame.K_KP_MINUS: "slower",
                       pygame.K_RIGHTBRACKET: "faster", pygame.K_EQUALS: "faster", pygame.K_PLUS: "faster",
                       pygame.K_KP_PLUS: "faster", pygame.K_h: "help", pygame.K_SLASH: "help", pygame.K_QUESTION: "help"}
            if key in actions:
                self._activate(actions[key])
            elif key in (pygame.K_LEFT, pygame.K_RIGHT):
                increment = self.clock.tick_hz if getattr(event, "mod", 0) & pygame.KMOD_SHIFT else 1
                self._seek(math.floor(self.clock.tick_position) + (-increment if key == pygame.K_LEFT else increment))
            elif key in (pygame.K_HOME, pygame.K_END):
                self._seek(0 if key == pygame.K_HOME else self.clock.final_tick)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            marker = self._marker_at(event.pos)
            if marker is not None:
                self._seek(marker)
            elif self.scrubber_rect.collidepoint(event.pos):
                self.dragging = True
                self._scrub(event.pos[0])
            else:
                for action, rect in self.buttons.items():
                    if rect.collidepoint(event.pos):
                        self._activate(action)
                        break
        elif event.type == pygame.MOUSEMOTION:
            self.hover_event = self._marker_at(event.pos)
            if self.dragging:
                self._scrub(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
            self._scrub(event.pos[0])
            self.dragging = False
        elif event.type == pygame.WINDOWFOCUSLOST:
            self.dragging = False
        return True

    def draw(self, sampled: Mapping[str, Any] | None = None) -> None:
        import pygame

        self.resize(self.surface)
        sampled = sample_replay(self.data, self.clock) if sampled is None else sampled
        frame = sampled["frame"]
        result = self.data["result"]
        self.surface.blit(self._background, (0, 0))
        old_clip = self.surface.get_clip()
        self.surface.set_clip(self.track_rect)
        self._draw_car(pygame, sampled["pose"], bool(frame["off_track"]))
        self.surface.set_clip(old_clip)
        pygame.draw.rect(self.surface, MUTED, self.track_rect, 1)
        pygame.draw.rect(self.surface, PANEL, self.panel_rect, border_radius=6)

        self._text(f"SYNAPSE / {self.track['meta']['display_name']}", pygame.Rect(16, 9, self.width - 32, 25), small=False)
        self._text(f"{self.data['source']['participant_id']}  |  {self.data['source']['mode']}  |  Read-only replay",
                   pygame.Rect(16, 35, self.width - 32, 20), color=MUTED)
        timing = f"lap {result['lap_time_s']:.3f} s" if result["lap_time_s"] is not None else f"elapsed {result['elapsed_s']:.3f} s; no lap"
        self._text(f"Recorded final: {RESULT_LABELS[result['status']]} | {timing} | at tick {result['steps']}",
                   pygame.Rect(16, 57, self.width - 32, 20), color=GOLD)

        current_status = RESULT_LABELS[result["status"]] if frame["tick"] == result["steps"] else "IN PROGRESS"
        if self.show_help:
            lines = ["KEYBOARD / ? to close", "Space: play / pause", "R: restart and play", "Left / Right: one tick",
                     "Shift + Left / Right: 1 s", "Home / End: endpoints", "[ / ] or - / +: speed",
                     "P / N: prev / next event", "H / ?: keyboard help", "Esc / Q: close viewer", "Seeking pauses playback."]
            for index, line in enumerate(lines):
                self._text(line, pygame.Rect(self.panel_rect.x + 10, self.panel_rect.y + 10 + index * 18,
                                            self.panel_rect.width - 20, 18), color=TEXT if index == 0 else MUTED)
        else:
            lines = [f"TICK {frame['tick']} / {result['steps']}",
                     f"Time {frame['tick'] / self.data['tick_hz']:.3f} / {result['elapsed_s']:.3f} s",
                     "OFF TRACK" if frame["off_track"] else "ON TRACK", current_status,
                     f"Speed    {frame['speed']:.2f} m/s", f"Lateral  {frame['lateral_velocity']:.2f} m/s",
                     f"Yaw       {frame['yaw_rate']:.3f} rad/s", f"Steer     {frame['steer']:.3f} rad",
                     f"Command {frame['steering_command']:+.3f}", f"Drive       {frame['throttle_brake']:+.3f}"]
            spacing = min(27, (self.panel_rect.height - 20) // len(lines))
            for index, line in enumerate(lines):
                color = WARNING if index == 2 and frame["off_track"] else TEXT
                if index == 3 and frame["tick"] == result["steps"]:
                    color = GOLD if result["status"] == "finished" else WARNING
                self._text(line, pygame.Rect(self.panel_rect.x + 10, self.panel_rect.y + 10 + index * spacing,
                                            self.panel_rect.width - 20, spacing), small=index not in (0, 2, 3), color=color)

        playback = "Playing" if self.clock.playing else "End of recording" if self.clock.tick_position >= self.clock.final_tick else "Paused"
        self._text(f"{playback} | Tick {frame['tick']} | {frame['tick'] / self.data['tick_hz']:.3f} / {result['elapsed_s']:.3f} s",
                   pygame.Rect(20, self.footer_rect.top + 4, self.width - 40, 20), color=MUTED)
        pygame.draw.circle(self.surface, ACCENT, (self._tick_x(sampled["tick_position"]), self.scrubber_rect.top + 8), 6)
        event_tick = frame["tick"] if self.hover_event is None else self.hover_event
        events = self.data["frames"][event_tick]["events"]
        event_text = ", ".join(EVENT_LABELS[event] for event in events) or ("none" if self.event_ticks else "none in this recording")
        self._text(f"{'Marker' if self.hover_event is not None else 'Events'} @ tick {event_tick}: {event_text}",
                   pygame.Rect(20, self.footer_rect.top + 63, self.width - 40, 20))
        labels = {"play": "Pause" if self.clock.playing else "Play", "restart": "Restart", "previous_event": "Prev event",
                  "next_event": "Next event", "slower": "-", "faster": "+", "speed": f"{self.clock.speed:g}x", "help": "Keys ?"}
        for action, rect in self.buttons.items():
            enabled = self._enabled(action)
            if action != "speed":
                pygame.draw.rect(self.surface, (31, 49, 67) if enabled else PANEL, rect, border_radius=4)
                pygame.draw.rect(self.surface, ACCENT if enabled else (54, 67, 80), rect, 1, border_radius=4)
            self._text(labels[action], rect.inflate(-4, 0), centered=True, color=TEXT if enabled or action == "speed" else MUTED)
        self._text("Space: play | Arrows: tick | Shift: 1 s | Home/End | [ ]: speed | P/N: events | ?: help",
                   pygame.Rect(16, self.footer_rect.top + 134, self.width - 32, 20), color=MUTED)


def run_pygame(
    replay_path: str | Path,
    *,
    size: tuple[int, int] = (1280, 800),
    speed: float = 1.0,
    headless: bool = False,
    start_seconds: float = 0.0,
    max_frames: int | None = None,
    screenshot: str | Path | None = None,
) -> None:
    """Run a resizable viewer, or save one deterministic paused headless frame."""
    if screenshot is not None:
        screenshot = Path(screenshot)
        if screenshot.suffix.lower() != ".png" or screenshot.resolve() == Path(replay_path).resolve():
            raise ValueError("screenshot must be a .png distinct from the recording")
        headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    data = load_replay(replay_path)
    clock = ReplayClock(data["tick_hz"], data["frames"][-1]["tick"], seconds=start_seconds, speed=speed, playing=not headless)
    pygame.init()
    try:
        surface = pygame.display.set_mode(size, pygame.RESIZABLE)
        renderer = PygameReplayRenderer(data, surface, clock)
        if headless:
            renderer.draw()
            pygame.display.flip()
            if screenshot is not None:
                pygame.image.save(renderer.surface, str(screenshot))
            return
        frame_count = 0
        running = True
        wall_clock = pygame.time.Clock()
        while running:
            # Advance before input so seeks/restart render their exact target,
            # without consuming a stale wall-time delta after the user's action.
            clock.advance(wall_clock.tick(60) / 1000.0)
            for event in pygame.event.get():
                if not renderer.handle_event(event):
                    running = False
                    break
            if not running:
                break
            renderer.draw()
            pygame.display.flip()
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break
    finally:
        pygame.quit()


__all__ = ["PygameReplayRenderer", "run_pygame"]
