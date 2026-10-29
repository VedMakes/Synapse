"""Generate complete full-lap Schema 1.1 dual-car replays for Monaco and Spa."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from racing_event.config import SimulationConfig
from racing_event.env import RacingEnv
from racing_event.track import TrackGeometry
from racing_event.trajectory import (
    EVENTS,
    ReplayWriter,
    replay_frame,
    replay_header,
    replay_result,
)


def run_full_lap_replay(
    track_path: Path,
    output_path: Path,
    participant_id: str,
    lateral_start: float,
    offset_bias: float,
    speed_factor: float,
    max_ticks: int,
) -> None:
    track = TrackGeometry.from_file(track_path)
    cfg = SimulationConfig()
    cfg = dataclasses.replace(
        cfg, episode=dataclasses.replace(cfg.episode, max_steps=max_ticks)
    )
    env = RacingEnv(track, cfg)
    seed = 42 if "beta" in participant_id else 7
    env.reset(
        seed=seed,
        spawn_s=track.start_line_s,
        lateral_offset=lateral_start,
        initial_speed=0.0,
        heading_error=0.0,
    )

    header = replay_header(track, cfg, participant_id=participant_id, mode="scripted")

    def get_action(env_inst: RacingEnv) -> np.ndarray:
        proj = env_inst.track.project(
            np.array([env_inst.state.x, env_inst.state.y], dtype=np.float64)
        )
        v = env_inst.state.speed
        lookaheads = np.linspace(4.0, max(40.0, v * 1.8), 24)
        target_v = 30.0 * speed_factor
        for dist in lookaheads:
            curv = abs(env_inst.track.curvature_at_s(proj.s + dist))
            corner_v = np.sqrt(16.0 / max(curv, 0.0006)) * speed_factor
            if v > corner_v:
                needed_dist = (v**2 - corner_v**2) / (2.0 * 11.0)
                if dist <= needed_dist * 1.2:
                    target_v = min(target_v, corner_v)
            else:
                if dist < 8.0:
                    target_v = min(target_v, corner_v)
        target_v = np.clip(target_v, 14.0, 30.0 * speed_factor)
        speed_err = target_v - v
        if speed_err > 0.5:
            throttle = 1.0 if speed_err > 3.0 else np.clip(speed_err / 2.5, 0.45, 1.0)
        elif speed_err < -0.8:
            throttle = np.clip(speed_err / 2.5, -1.0, -0.3)
        else:
            throttle = 0.2

        la_dist = max(7.0, v * 0.35)
        la_s = proj.s + la_dist
        la_curv = env_inst.track.curvature_at_s(la_s)
        desired_angle = np.arctan(env_inst.config.car.wheelbase_m * la_curv)
        steering = desired_angle / env_inst.config.car.max_steering_rad
        lat_err = proj.lateral_offset - offset_bias
        steering -= 0.65 * np.clip(lat_err / 3.0, -1.0, 1.0)
        steering -= 0.75 * env_inst.track.observation_heading_error(
            env_inst.state.heading, proj
        )
        return np.clip(np.asarray([steering, throttle], dtype=np.float64), -1.0, 1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ReplayWriter(output_path, header) as writer:
        edge = track.edge_status(
            np.array([env.state.x, env.state.y]),
            float(env.state.heading),
            cfg.car.length_m,
            cfg.car.width_m,
        )
        s0 = {
            "x": env.state.x,
            "y": env.state.y,
            "heading": env.state.heading,
            "speed": env.state.speed,
            "lateral_velocity": env.state.lateral_velocity,
            "yaw_rate": env.state.yaw_rate,
            "steer": env.state.steering_angle,
        }
        writer.append(replay_frame(0, s0, off_track=bool(edge.off_track)))

        status = "dnf_timeout"
        lap_time_s = None
        final_steps = 0

        for tick in range(1, max_ticks + 1):
            act = get_action(env)
            _, _, term, trunc, info = env.step(act)
            final_steps = tick
            edge = track.edge_status(
                np.array([env.state.x, env.state.y]),
                float(env.state.heading),
                cfg.car.length_m,
                cfg.car.width_m,
            )

            raw_events = list(info.get("events", []))
            if info.get("lap_complete", False) and "lap_complete" not in raw_events:
                raw_events.append("lap_complete")
            valid_events = [e for e in dict.fromkeys(raw_events) if e in EVENTS]

            st = info.get("state", {})
            s_dict = {
                "x": st.get("x", env.state.x),
                "y": st.get("y", env.state.y),
                "heading": st.get("heading", env.state.heading),
                "speed": st.get("speed", env.state.speed),
                "lateral_velocity": st.get(
                    "lateral_velocity", env.state.lateral_velocity
                ),
                "yaw_rate": st.get("yaw_rate", env.state.yaw_rate),
                "steer": st.get("steer", env.state.steering_angle),
            }
            fr = replay_frame(
                tick,
                s_dict,
                steering_command=float(act[0]),
                throttle_brake=float(act[1]),
                off_track=bool(edge.off_track),
                events=valid_events,
                terminated=bool(term),
                truncated=bool(trunc),
            )
            writer.append(fr)
            if info.get("lap_complete", False):
                status = "finished"
                lap_time_s = tick / cfg.tick_hz
                break
            if term or trunc:
                status = "dnf_crash" if info.get("crash") else "dnf_timeout"
                break

        res = SimpleNamespace(
            status=status,
            steps=final_steps,
            lap_time_s=lap_time_s,
            trajectory_hash=hashlib.sha256(
                f"{participant_id}_{final_steps}".encode("ascii")
            ).hexdigest(),
        )
        writer.finish(replay_result(res, cfg.tick_hz))
        print(
            f"Successfully generated {output_path.name}: {status} in {final_steps} steps ({final_steps / 30:.2f}s)"
        )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tracks_dir = repo_root / "tracks" / "train"
    renderer_dir = repo_root / "renderer"

    print("Generating full-lap Monaco replays...")
    run_full_lap_replay(
        tracks_dir / "monaco.json",
        renderer_dir / "monaco_a.json",
        participant_id="team_alpha",
        lateral_start=-1.2,
        offset_bias=-0.4,
        speed_factor=1.0,
        max_ticks=5500,
    )
    run_full_lap_replay(
        tracks_dir / "monaco.json",
        renderer_dir / "monaco_b.json",
        participant_id="team_beta",
        lateral_start=1.2,
        offset_bias=0.4,
        speed_factor=0.985,
        max_ticks=5500,
    )

    print("Generating full-lap Spa replays...")
    run_full_lap_replay(
        tracks_dir / "spa.json",
        renderer_dir / "spa_a.json",
        participant_id="team_alpha",
        lateral_start=-1.2,
        offset_bias=-0.4,
        speed_factor=1.0,
        max_ticks=9500,
    )
    run_full_lap_replay(
        tracks_dir / "spa.json",
        renderer_dir / "spa_b.json",
        participant_id="team_beta",
        lateral_start=1.2,
        offset_bias=0.4,
        speed_factor=0.985,
        max_ticks=9500,
    )
    print("All replays successfully generated!")


if __name__ == "__main__":
    main()

