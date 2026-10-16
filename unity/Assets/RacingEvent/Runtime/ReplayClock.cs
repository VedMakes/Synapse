using System;
using UnityEngine;

namespace RacingEvent.Unity
{
    public readonly struct ReplayPoseSample
    {
        public readonly double time_s;
        public readonly int tick;
        public readonly double alpha;
        public readonly ReplayFrame frame;
        public readonly double x;
        public readonly double y;
        public readonly double heading;

        public ReplayPoseSample(double time, int completedTick, double interpolation, ReplayFrame completedFrame, double positionX, double positionY, double angle)
        {
            time_s = time;
            tick = completedTick;
            alpha = interpolation;
            frame = completedFrame;
            x = positionX;
            y = positionY;
            heading = angle;
        }
    }

    public sealed class ReplayClock
    {
        public readonly ReplayDocument Replay;
        public double TimeSeconds { get; private set; }
        public bool Playing { get; private set; }
        public double Speed { get; private set; } = 1;
        public double DurationSeconds => Replay.frames[Replay.frames.Length - 1].tick / (double)Replay.tick_hz;

        public ReplayClock(ReplayDocument replay)
        {
            if (replay == null || replay.frames == null || replay.frames.Length == 0 || replay.tick_hz <= 0) throw new ArgumentException("Replay clock requires a validated replay.", nameof(replay));
            Replay = replay;
        }

        public void Advance(double unscaledDeltaSeconds)
        {
            if (!Playing) return;
            if (double.IsNaN(unscaledDeltaSeconds) || double.IsInfinity(unscaledDeltaSeconds) || unscaledDeltaSeconds < 0) throw new ArgumentOutOfRangeException(nameof(unscaledDeltaSeconds));
            TimeSeconds = Math.Min(DurationSeconds, TimeSeconds + unscaledDeltaSeconds * Speed);
            if (TimeSeconds >= DurationSeconds) Playing = false;
        }

        public void Play()
        {
            if (DurationSeconds <= 0) return;
            if (TimeSeconds >= DurationSeconds) TimeSeconds = 0;
            Playing = true;
        }

        public void Pause() => Playing = false;

        public void Toggle()
        {
            if (Playing) Pause();
            else Play();
        }

        public void Restart()
        {
            TimeSeconds = 0;
            Playing = false;
        }

        public void SetSpeed(double speed)
        {
            if (double.IsNaN(speed) || double.IsInfinity(speed) || speed <= 0) throw new ArgumentOutOfRangeException(nameof(speed));
            Speed = speed;
        }

        public void Seek(double timeSeconds)
        {
            if (double.IsNaN(timeSeconds) || double.IsInfinity(timeSeconds)) throw new ArgumentOutOfRangeException(nameof(timeSeconds));
            TimeSeconds = Math.Max(0, Math.Min(DurationSeconds, timeSeconds));
            if (TimeSeconds >= DurationSeconds) Playing = false;
        }

        public void SetTick(int tick)
        {
            if (tick < 0 || tick > Replay.frames[Replay.frames.Length - 1].tick) throw new ArgumentOutOfRangeException(nameof(tick));
            TimeSeconds = tick / (double)Replay.tick_hz;
            Playing = false;
        }

        public ReplayPoseSample Sample() => ReplaySampling.SampleAtTime(Replay, TimeSeconds);
    }

    public static class ReplaySampling
    {
        public static ReplayPoseSample SampleAtTime(ReplayDocument replay, double timeSeconds)
        {
            if (replay == null || replay.frames == null || replay.frames.Length == 0 || replay.tick_hz <= 0) throw new ArgumentException("Replay sampling requires a validated replay.", nameof(replay));
            var lastTick = replay.frames[replay.frames.Length - 1].tick;
            var duration = lastTick / (double)replay.tick_hz;
            var time = Math.Max(0, Math.Min(duration, timeSeconds));
            var position = time >= duration ? lastTick : time * replay.tick_hz;
            var nearest = Math.Round(position);
            if (Math.Abs(position - nearest) <= 8 * double.Epsilon * Math.Max(1, Math.Abs(position))) position = nearest;
            var tick = Math.Max(0, Math.Min(lastTick, (int)Math.Floor(position)));
            var nextTick = Math.Min(tick + 1, lastTick);
            var alpha = tick == lastTick ? 0 : position - tick;
            var frame = replay.frames[tick];
            var next = replay.frames[nextTick];
            return new ReplayPoseSample(
                time,
                tick,
                alpha,
                frame,
                alpha == 0 ? frame.x : Lerp(frame.x, next.x, alpha),
                alpha == 0 ? frame.y : Lerp(frame.y, next.y, alpha),
                alpha == 0 ? frame.heading : InterpolateHeading(frame.heading, next.heading, alpha));
        }

        public static double InterpolateHeading(double from, double to, double alpha)
        {
            if (alpha <= 0) return from;
            if (alpha >= 1) return to;
            var difference = (to % (2 * Math.PI)) - (from % (2 * Math.PI));
            var shortest = Math.Atan2(Math.Sin(difference), Math.Cos(difference));
            return from + shortest * alpha;
        }

        public static float UnityYawDegrees(double pythonHeading)
        {
            return (float)(90 - pythonHeading * Mathf.Rad2Deg);
        }

        private static double Lerp(double from, double to, double alpha) => from + (to - from) * alpha;
    }
}
