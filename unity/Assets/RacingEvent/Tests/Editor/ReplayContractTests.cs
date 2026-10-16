using NUnit.Framework;
using RacingEvent.Unity;

namespace RacingEvent.Tests
{
    public sealed class ReplayContractTests
    {
        [Test]
        public void PythonHeadingMapsToDocumentedUnityYaw()
        {
            Assert.That(ReplaySampling.UnityYawDegrees(0), Is.EqualTo(90f).Within(0.0001f));
            Assert.That(ReplaySampling.UnityYawDegrees(System.Math.PI / 2), Is.EqualTo(0f).Within(0.0001f));
        }

        [Test]
        public void HeadingInterpolationUsesTheShortestArc()
        {
            var from = 179 * System.Math.PI / 180;
            var to = -179 * System.Math.PI / 180;
            Assert.That(ReplaySampling.InterpolateHeading(from, to, 0.5), Is.EqualTo(System.Math.PI).Within(0.0001));
        }

        [Test]
        public void ClockSetsExactTicksAndDoesNotExtrapolatePastTheEnd()
        {
            var replay = new ReplayDocument
            {
                tick_hz = 30,
                frames = new[]
                {
                    new ReplayFrame { tick = 0, x = 1, y = 2, heading = 0 },
                    new ReplayFrame { tick = 1, x = 3, y = 4, heading = 1 }
                }
            };
            var clock = new ReplayClock(replay);
            clock.SetTick(1);
            Assert.That(clock.TimeSeconds, Is.EqualTo(1d / 30d).Within(0.000000001));
            Assert.That(clock.Sample().x, Is.EqualTo(3));
            clock.Play();
            clock.Advance(10);
            Assert.That(clock.Sample().x, Is.EqualTo(3));
            Assert.That(clock.Playing, Is.False);
        }

        [Test]
        public void ReaderRejectsWrongSchemaBeforeAnyRendererWork()
        {
            Assert.Throws<ReplayFormatException>(() => ReplayReader.Read("{\"schema_version\":\"1.0\"}"));
        }
    }
}
