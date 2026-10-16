using System;
using System.Collections.Generic;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace RacingEvent.Unity
{
    [Serializable]
    public sealed class ReplayDocument
    {
        public string schema_version;
        public string kind;
        public ReplayEngine engine;
        public ReplaySource source;
        public int tick_hz;
        public ReplayCar car;
        public ReplayTrack track;
        public ReplayFrame[] frames;
        public ReplayResult result;
        public string payload_hash;
    }

    [Serializable]
    public sealed class ReplayEngine
    {
        public string code_hash;
        public string config_hash;
    }

    [Serializable]
    public sealed class ReplaySource
    {
        public string mode;
        public string participant_id;
    }

    [Serializable]
    public sealed class ReplayCar
    {
        public double length_m;
        public double width_m;
    }

    [Serializable]
    public sealed class ReplayTrack
    {
        public string version;
        public double[][] centerline;
        public double[] s;
        public double[] heading;
        public double[] curvature;
        public double[] half_width_left;
        public double[] half_width_right;
        public double[][] left_boundary;
        public double[][] right_boundary;
        public double[][] barrier_left;
        public double[][] barrier_right;
        public double L;
        public double runoff_width;
        public double start_line_s;
        public ReplayTrackMeta meta;
    }

    [Serializable]
    public sealed class ReplayTrackMeta
    {
        public string id;
        public string display_name;
        public string source_note;
        public string geometry_version;
        public double car_width_m;
        public double track_width_m;
        public double width_ratio;
        public double minimum_radius_m;
        public string content_hash;

        // Keep future non-authoritative metadata available without making it
        // part of the runtime contract.
        [JsonExtensionData]
        public IDictionary<string, JToken> extra;
    }

    [Serializable]
    public sealed class ReplayFrame
    {
        public int tick;
        public double x;
        public double y;
        public double heading;
        public double speed;
        public double lateral_velocity;
        public double yaw_rate;
        public double steer;
        public double steering_command;
        public double throttle_brake;
        public bool off_track;
        public string[] events;
        public bool terminated;
        public bool truncated;
    }

    [Serializable]
    public sealed class ReplayResult
    {
        public string status;
        public int steps;
        public double? lap_time_s;
        public double elapsed_s;
        public string trajectory_hash;
    }
}
