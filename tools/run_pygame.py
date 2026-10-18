"""Play a validated SYNAPSE replay with the read-only Pygame renderer."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racing_event.pygame_renderer import MIN_WINDOW_SIZE, run_pygame


def parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("size must look like WIDTHxHEIGHT") from exc
    if width < MIN_WINDOW_SIZE[0] or height < MIN_WINDOW_SIZE[1]:
        raise argparse.ArgumentTypeError("size must be at least 640x480")
    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--size", type=parse_size, default=(1280, 800))
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true", help="render one frame with SDL dummy drivers")
    parser.add_argument("--screenshot", type=Path, help="save one deterministic paused PNG at --start-seconds (implies --headless)")
    parser.add_argument("--max-frames", type=int, default=None, help="stop after this many interactive frames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0 or args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--speed and --max-frames must be positive")
    try:
        run_pygame(
            args.replay,
            size=args.size,
            speed=args.speed,
            headless=args.headless,
            start_seconds=args.start_seconds,
            max_frames=args.max_frames,
            screenshot=args.screenshot,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"pygame replay failed: {exc}") from exc


if __name__ == "__main__":
    main()
