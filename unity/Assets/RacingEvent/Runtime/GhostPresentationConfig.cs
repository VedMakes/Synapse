using UnityEngine;

namespace RacingEvent.Unity
{
    public enum GhostRevealMode
    {
        Simultaneous,
        Staggered
    }

    [CreateAssetMenu(menuName = "Racing Event/Ghost Presentation Config", fileName = "GhostPresentationConfig")]
    public sealed class GhostPresentationConfig : ScriptableObject
    {
        [Header("Placement")]
        public float visual_height = 0.08f;
        public float visual_scale = 1f;

        [Header("Presentation")]
        public GhostRevealMode reveal_mode = GhostRevealMode.Simultaneous;
        [Min(0f)] public float reveal_delay_seconds;
        public bool show_label = true;
        public bool show_trail;
        [Min(0f)] public float trail_length_seconds = 2f;

        [Header("Recorded effects")]
        public bool enable_event_effects = true;
        public bool smoke_on_off_track = true;
        public bool impact_on_barrier = true;
        public bool confetti_on_lap_complete = true;

        public bool ShouldReveal(double elapsed_seconds)
        {
            return reveal_mode == GhostRevealMode.Simultaneous || elapsed_seconds >= reveal_delay_seconds;
        }
    }
}
