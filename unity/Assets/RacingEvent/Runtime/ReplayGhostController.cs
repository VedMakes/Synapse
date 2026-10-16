using System;
using UnityEngine;

namespace RacingEvent.Unity
{
    public sealed class ReplayGhostController : MonoBehaviour
    {
        [SerializeField] private TextAsset replay_asset;
        [SerializeField] private Transform car_visual;
        [SerializeField] private GhostPresentationConfig presentation;

        public ReplayDocument Replay { get; private set; }
        public ReplayClock Clock { get; private set; }

        // VECTOR-owned effects can subscribe to recorded events. The event
        // payload is read-only presentation input; it never returns an action.
        public event Action<ReplayFrame, string> RecordedEvent;

        private int lastEventTick = -1;
        private Vector3 initialScale = Vector3.one;

        private void Awake()
        {
            if (replay_asset == null) throw new InvalidOperationException("ReplayGhostController needs a schema 1.1 replay TextAsset.");
            Replay = ReplayReader.Read(replay_asset.text);
            Clock = new ReplayClock(Replay);
            if (car_visual == null) car_visual = transform;
            initialScale = car_visual.localScale;
            ApplyPose(Clock.Sample());
        }

        private void Update()
        {
            if (Clock == null || car_visual == null) return;
            Clock.Advance(Time.unscaledDeltaTime);
            if (presentation != null && !presentation.ShouldReveal(Clock.TimeSeconds))
            {
                car_visual.gameObject.SetActive(false);
                return;
            }

            car_visual.gameObject.SetActive(true);
            var sample = Clock.Sample();
            ApplyPose(sample);
            EmitRecordedEvents(sample);
        }

        public void SetTick(int tick)
        {
            if (Clock == null) throw new InvalidOperationException("Replay has not been loaded.");
            Clock.SetTick(tick);
            lastEventTick = tick - 1;
            ApplyPose(Clock.Sample());
        }

        private void ApplyPose(ReplayPoseSample sample)
        {
            var height = presentation == null ? 0.08f : presentation.visual_height;
            var scale = presentation == null ? 1f : presentation.visual_scale;
            car_visual.position = new Vector3((float)sample.x, height, (float)sample.y);
            car_visual.rotation = Quaternion.Euler(0, ReplaySampling.UnityYawDegrees(sample.heading), 0);
            car_visual.localScale = initialScale * scale;
        }

        private void EmitRecordedEvents(ReplayPoseSample sample)
        {
            if (sample.tick <= lastEventTick || sample.frame.events == null) return;
            lastEventTick = sample.tick;
            if (presentation != null && !presentation.enable_event_effects) return;
            foreach (var eventName in sample.frame.events)
            {
                RecordedEvent?.Invoke(sample.frame, eventName);
            }
        }
    }
}
