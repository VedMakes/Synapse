using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace RacingEvent.Unity
{
    public sealed class ReplayFormatException : Exception
    {
        public ReplayFormatException(string message) : base(message) { }
    }

    public static class ReplayReader
    {
        private static readonly HashSet<string> EventNames = new HashSet<string>
        {
            "off_track_start", "off_track_end", "barrier_contact", "lap_complete"
        };

        private static readonly HashSet<string> ResultStatuses = new HashSet<string>
        {
            "finished", "dnf_crash", "dnf_stuck", "dnf_backward", "dnf_timeout"
        };

        private static readonly JsonSerializerSettings SerializerSettings = new JsonSerializerSettings
        {
            MissingMemberHandling = MissingMemberHandling.Error,
            FloatParseHandling = FloatParseHandling.Double,
            DateParseHandling = DateParseHandling.None,
            MetadataPropertyHandling = MetadataPropertyHandling.Ignore,
            TypeNameHandling = TypeNameHandling.None
        };

        public static ReplayDocument Read(string json)
        {
            if (string.IsNullOrWhiteSpace(json)) throw new ReplayFormatException("Replay JSON is empty.");

            JObject root;
            try
            {
                root = JObject.Parse(json, new JsonLoadSettings
                {
                    DuplicatePropertyNameHandling = DuplicatePropertyNameHandling.Error
                });
            }
            catch (Exception exception) when (exception is JsonException || exception is ArgumentException)
            {
                throw new ReplayFormatException("Replay JSON is malformed: " + exception.Message);
            }

            RequireFields(root, new[] { "schema_version", "kind", "engine", "source", "tick_hz", "car", "track", "frames", "result", "payload_hash" }, "replay");
            ReplayDocument replay;
            try
            {
                replay = JsonConvert.DeserializeObject<ReplayDocument>(root.ToString(Formatting.None), SerializerSettings);
            }
            catch (JsonException exception)
            {
                throw new ReplayFormatException("Replay fields are malformed: " + exception.Message);
            }

            if (replay == null) throw new ReplayFormatException("Replay document is null.");
            ValidateReplay(replay, root);
            return replay;
        }

        public static void ValidateReplay(ReplayDocument replay, JObject source)
        {
            Require(replay.schema_version == "1.1", "Unsupported replay schema; expected 1.1.");
            Require(replay.kind == "racing_event_replay", "Invalid replay kind.");
            Require(replay.tick_hz > 0, "Replay tick_hz must be positive.");
            RequireHash(replay.engine == null ? null : replay.engine.code_hash, "engine.code_hash");
            RequireHash(replay.engine == null ? null : replay.engine.config_hash, "engine.config_hash");
            Require(replay.source != null && (replay.source.mode == "scripted" || replay.source.mode == "evaluation"), "Invalid replay source mode.");
            Require(replay.source.participant_id != null && replay.source.participant_id.Length >= 1 && replay.source.participant_id.Length <= 128, "Invalid participant id.");
            Require(replay.car != null && PositiveFinite(replay.car.length_m) && PositiveFinite(replay.car.width_m), "Invalid car dimensions.");
            ValidateTrack(replay.track, (JObject)source["track"]);
            RequireHash(replay.payload_hash, "payload_hash");
            Require(replay.frames != null && replay.frames.Length >= 2, "Replay needs an initial frame and one transition.");

            for (var index = 0; index < replay.frames.Length; index++)
            {
                ValidateFrame(replay.frames[index], index);
                if (index < replay.frames.Length - 1)
                {
                    Require(!replay.frames[index].terminated && !replay.frames[index].truncated, "Terminal frame must be last.");
                }
            }

            ValidateResult(replay.result, replay.frames[replay.frames.Length - 1], replay.tick_hz);
            var calculatedTrackHash = ComputeTrackContentHash((JObject)source["track"]);
            Require(string.Equals(calculatedTrackHash, replay.track.meta.content_hash, StringComparison.Ordinal), "Embedded track content hash mismatch.");
        }

        private static void ValidateTrack(ReplayTrack track, JObject source)
        {
            Require(track != null && source != null && track.version == "1.0", "Unsupported or missing track version.");
            Require(track.centerline != null && track.centerline.Length >= 8, "Track needs at least eight centerline samples.");
            var count = track.centerline.Length;
            foreach (var points in new[] { track.left_boundary, track.right_boundary, track.barrier_left, track.barrier_right })
            {
                Require(points != null && points.Length == count, "Track boundary arrays must match the centreline.");
                foreach (var point in points) Require(point != null && point.Length == 2 && Finite(point[0]) && Finite(point[1]), "Track contains an invalid coordinate.");
            }
            foreach (var values in new[] { track.s, track.heading, track.curvature, track.half_width_left, track.half_width_right })
            {
                Require(values != null && values.Length == count && values.All(Finite), "Track scalar arrays must match the centreline and be finite.");
            }
            Require(PositiveFinite(track.L) && track.runoff_width >= 0 && Finite(track.runoff_width) && track.start_line_s >= 0 && track.start_line_s < track.L, "Invalid track dimensions.");
            Require(track.meta != null && !string.IsNullOrEmpty(track.meta.id) && !string.IsNullOrEmpty(track.meta.display_name), "Missing track metadata.");
            RequireHash(track.meta.content_hash, "track.meta.content_hash");
        }

        private static void ValidateFrame(ReplayFrame frame, int expectedTick)
        {
            Require(frame != null && frame.tick == expectedTick, "Frames must have consecutive ticks starting at zero.");
            foreach (var value in new[] { frame.x, frame.y, frame.heading, frame.speed, frame.lateral_velocity, frame.yaw_rate, frame.steer, frame.steering_command, frame.throttle_brake }) Require(Finite(value), "Frame contains a non-finite value.");
            Require(frame.speed >= 0 && Math.Abs(frame.steering_command) <= 1 && Math.Abs(frame.throttle_brake) <= 1, "Frame contains an invalid speed or applied control.");
            Require(frame.events != null && frame.events.Distinct().Count() == frame.events.Length && frame.events.All(EventNames.Contains), "Frame contains an unsupported or duplicate event.");
            Require(!(frame.terminated && frame.truncated), "Frame cannot be both terminated and truncated.");
            if (expectedTick == 0) Require(frame.events.Length == 0 && !frame.terminated && !frame.truncated, "Initial frame cannot contain events or terminal flags.");
            if (frame.events.Contains("barrier_contact") || frame.events.Contains("lap_complete")) Require(frame.terminated, "Terminal events require recorded termination.");
            if (frame.events.Contains("off_track_start")) Require(frame.off_track && !frame.events.Contains("off_track_end"), "off_track_start disagrees with surface state.");
            if (frame.events.Contains("off_track_end")) Require(!frame.off_track, "off_track_end disagrees with surface state.");
        }

        private static void ValidateResult(ReplayResult result, ReplayFrame last, int tickHz)
        {
            Require(result != null && ResultStatuses.Contains(result.status), "Invalid replay result.");
            Require(result.steps == last.tick && result.steps >= 1, "Result steps do not match the final frame.");
            var duration = result.steps / (double)tickHz;
            Require(Finite(result.elapsed_s) && Math.Abs(result.elapsed_s - duration) <= 1e-9, "Result elapsed time is not fixed-step time.");
            RequireHash(result.trajectory_hash, "trajectory_hash");
            if (result.status == "finished")
            {
                Require(last.terminated && last.events.Contains("lap_complete") && !last.events.Contains("barrier_contact"), "Finish result disagrees with the final frame.");
                Require(result.lap_time_s.HasValue && Math.Abs(result.lap_time_s.Value - duration) <= 1e-9, "Finish lap time is not fixed-step time.");
            }
            else
            {
                Require(!result.lap_time_s.HasValue, "DNF lap time must be null.");
            }
            if (result.status == "dnf_crash") Require(last.terminated && last.events.Contains("barrier_contact"), "Crash result requires barrier contact.");
            if (result.status == "dnf_stuck") Require(last.truncated, "Stuck result requires truncation.");
            if (result.status == "dnf_backward") Require(last.terminated, "Backward result requires termination.");
            if (result.status == "dnf_timeout") Require(!last.terminated, "Timeout cannot disguise termination.");
        }

        public static string ComputeTrackContentHash(JObject track)
        {
            if (track == null) throw new ReplayFormatException("Missing track object.");
            var clone = (JObject)track.DeepClone();
            var meta = clone["meta"] as JObject;
            if (meta == null) throw new ReplayFormatException("Missing track metadata.");
            meta.Remove("content_hash");
            using (var sha = SHA256.Create())
            {
                return BitConverter.ToString(sha.ComputeHash(Encoding.UTF8.GetBytes(CanonicalJson(clone)))).Replace("-", string.Empty).ToLowerInvariant();
            }
        }

        private static string CanonicalJson(JToken token)
        {
            if (token is JObject objectToken)
            {
                return "{" + string.Join(",", objectToken.Properties().OrderBy(property => property.Name, StringComparer.Ordinal).Select(property => JsonConvert.ToString(property.Name) + ":" + CanonicalJson(property.Value))) + "}";
            }
            if (token is JArray arrayToken) return "[" + string.Join(",", arrayToken.Select(CanonicalJson)) + "]";
            if (token.Type == JTokenType.String) return JsonConvert.ToString(token.Value<string>());
            if (token.Type == JTokenType.Boolean) return token.Value<bool>() ? "true" : "false";
            if (token.Type == JTokenType.Null) return "null";
            return ((JValue)token).ToString(CultureInfo.InvariantCulture);
        }

        private static void RequireFields(JObject objectToken, IEnumerable<string> expected, string name)
        {
            var actual = new HashSet<string>(objectToken.Properties().Select(property => property.Name));
            var required = new HashSet<string>(expected);
            Require(actual.SetEquals(required), "Invalid " + name + " fields.");
        }

        private static void RequireHash(string value, string name)
        {
            Require(!string.IsNullOrEmpty(value) && value.Length == 64 && value.All(character => "0123456789abcdef".IndexOf(character) >= 0), name + " must be a lowercase SHA-256 digest.");
        }

        private static bool Finite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
        private static bool PositiveFinite(double value) => Finite(value) && value > 0;
        private static void Require(bool condition, string message)
        {
            if (!condition) throw new ReplayFormatException(message);
        }
    }
}
