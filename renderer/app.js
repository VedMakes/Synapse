/*
 * Synapse 2.5D Race Renderer
 *
 * I built this real time WebGL broadcast renderer to recreate an authentic Formula 1 TV viewing experience.
 * Here is my overall architecture and approach:
 * 1. Data Ingestion: I parse custom or default telemetry JSON recordings and interpolate continuous positions and orientations at 60 frames per second.
 * 2. Web Audio Synthesis: I implemented a zero dependency procedural synthesizer recreating dual V6 turbo hybrid engine acoustics, dynamic RPM spooling, and crowd ambience.
 * 3. 3D World Generation: I generate realistic procedural circuits with asphalt bump maps, FIA curbs, safety barriers, stadium lighting, and spectator grandstands.
 * 4. F1 Vehicle Models: I model aerodynamic chassis geometry with clearcoat paint, carbon splitters, glowing brake discs, and an X ray overlap mode to prevent visual clipping.
 * 5. Broadcast Director: I switch between 6 cinematic highlight camera angles in hero mode, and provide battle chase, cockpit, top down, and trackside cameras during the main race.
 */

import * as THREE from 'three';
import { connectLive, parseLiveMessage } from './live_source.js';

/*
 * Phase 1: Data Ingestion and Telemetry Interpolation Engine
 * Here I handle parsing incoming race replays and calculating smooth continuous vectors between discrete simulation ticks.
 */

/*
 * I parse and validate the incoming Schema 1.1 replay JSON.
 * My approach extracts the participant metadata, tick frequency, car dimensions,
 * track boundary polylines, and frame arrays, raising clear errors if format mismatches occur.
 */
function parseReplay(data) {
  if (data.schema_version !== '1.1' || data.kind !== 'racing_event_replay') {
    throw new Error('Unsupported replay format: expected schema 1.1, racing_event_replay');
  }
  return {
    participantId: data.source?.participant_id || 'Unknown Racer',
    tickHz: data.tick_hz || 30,
    car: data.car || { length_m: 4.8, width_m: 2.0 },
    track: data.track,
    frames: data.frames,
    result: data.result || {},
  };
}

/*
 * Shortest angle linear interpolation for car heading.
 * I specifically designed this helper to eliminate the 360 degree spin flip bug.
 * When rotation crosses between positive pi and negative pi, naive lerping takes the long path.
 * By normalizing the angular difference between minus pi and plus pi, the car rotates smoothly along the shortest path.
 */
function lerpAngle(a, b, t) {
  let diff = b - a;
  diff = diff - Math.round(diff / (2 * Math.PI)) * 2 * Math.PI;
  return a + diff * t;
}

/*
 * Here I sample telemetry frames at an arbitrary continuous time in seconds.
 * I find the bounding simulation ticks and linearly interpolate position, velocity, heading,
 * throttle, brake, and steering so the visual playback runs at a rock solid 60 frames per second.
 */
function sampleAtTime(frames, tickHz, timeS) {
  if (!frames || frames.length === 0) return null;
  const lastTick = frames.length - 1;
  const duration = lastTick / tickHz;
  const clampedTime = Math.max(0, Math.min(timeS, duration));

  let pos = clampedTime * tickHz;
  const nearest = Math.round(pos);
  if (Math.abs(pos - nearest) < 1e-9) pos = nearest;

  const tickA = Math.min(Math.floor(pos), lastTick);
  const tickB = Math.min(tickA + 1, lastTick);
  const frac = tickA === tickB ? 0 : pos - tickA;

  const a = frames[tickA];
  const b = frames[tickB];

  return {
    x: a.x + (b.x - a.x) * frac,
    y: a.y + (b.y - a.y) * frac,
    heading: lerpAngle(a.heading, b.heading, frac),
    speed: a.speed + (b.speed - a.speed) * frac,
    steer: a.steer + (b.steer - a.steer) * frac,
    throttle_brake: a.throttle_brake + (b.throttle_brake - a.throttle_brake) * frac,
    yaw_rate: a.yaw_rate + (b.yaw_rate - a.yaw_rate) * frac,
    lateral_velocity: a.lateral_velocity + (b.lateral_velocity - a.lateral_velocity) * frac,
    off_track: a.off_track,
    events: a.events || [],
    terminated: a.terminated || false,
    truncated: a.truncated || false,
    tick: tickA,
    finished: clampedTime >= duration,
  };
}

/*
 * Audio Suite: Pure Web Audio API Real Time Synthesizer
 *
 * I chose to build a dedicated procedural audio synthesizer directly using the Web Audio API
 * instead of relying on external audio sample files. This provides instantaneous dynamic audio that
 * responds immediately to throttle, speed, braking, and camera focus without network latency.
 *
 * My sound system contains:
 * 1. Dual F1 Turbocharged V6 Hybrid Engines with realistic RPM spool and deceleration overrun pops
 * 2. Grandstand Crowd and Trackside Ambience with procedural pink noise and custom audio support
 * 3. Ambient Broadcast Music Bed with smooth chord progressions
 * 4. Dedicated Hero Section BGM Player with seamless fade transitions
 */

class F1EngineSound {
  constructor(ctx, dest, { pan = 0 } = {}) {
    this.ctx = ctx;
    this.panner = ctx.createStereoPanner ? ctx.createStereoPanner() : null;
    if (this.panner) this.panner.pan.value = pan;

    this.gain = ctx.createGain();
    this.gain.gain.value = 0.0;

    /* Resonant bandpass filter tracking calculated RPM */
    this.bpFilter = ctx.createBiquadFilter();
    this.bpFilter.type = 'bandpass';
    this.bpFilter.Q.value = 2.4;
    this.bpFilter.frequency.value = 850;

    /* Lowpass smoothing filter to tame harsh digital harmonics */
    this.lpFilter = ctx.createBiquadFilter();
    this.lpFilter.type = 'lowpass';
    this.lpFilter.frequency.value = 3200;

    /* Waveshaper distortion curve giving the raw V6 cylinder combustion roar */
    this.shaper = ctx.createWaveShaper();
    this.shaper.curve = this._makeDistortionCurve(16);
    this.shaper.oversample = '2x';

    /* Oscillators: Sawtooth for combustion firing plus triangle for engine block body */
    this.oscSaw = ctx.createOscillator();
    this.oscSaw.type = 'sawtooth';
    this.oscSaw.frequency.value = 120;

    this.oscSub = ctx.createOscillator();
    this.oscSub.type = 'triangle';
    this.oscSub.frequency.value = 60;

    /* High frequency sine oscillator simulating turbocharger spool whine */
    this.oscTurbo = ctx.createOscillator();
    this.oscTurbo.type = 'sine';
    this.oscTurbo.frequency.value = 2600;

    this.turboGain = ctx.createGain();
    this.turboGain.gain.value = 0.0;

    /* Audio node wiring and routing to destination */
    this.oscSaw.connect(this.shaper);
    this.oscSub.connect(this.shaper);
    this.shaper.connect(this.bpFilter);
    this.bpFilter.connect(this.lpFilter);
    this.lpFilter.connect(this.gain);

    this.oscTurbo.connect(this.turboGain);
    this.turboGain.connect(this.gain);

    if (this.panner) {
      this.gain.connect(this.panner);
      this.panner.connect(dest);
    } else {
      this.gain.connect(dest);
    }

    this.oscSaw.start();
    this.oscSub.start();
    this.oscTurbo.start();
  }

  _makeDistortionCurve(amount) {
    const k = amount;
    const n = 256;
    const curve = new Float32Array(n);
    const deg = Math.PI / 180;
    for (let i = 0; i < n; ++i) {
      const x = (i * 2) / n - 1;
      curve[i] = ((3 + k) * x * 20 * deg) / (Math.PI + k * Math.abs(x));
    }
    return curve;
  }

  update(sample, volumeMultiplier = 1.0) {
    if (!sample || !this.ctx) return;
    const speedKmh = Math.max(0, sample.speed * 3.6);
    const thr = Math.max(0, sample.throttle_brake);
    const brk = Math.max(0, -sample.throttle_brake);

    /* I calculate simulated gear ratios from 1 to 8 based on road speed */
    const gearSpeeds = [0, 75, 120, 165, 210, 255, 295, 335, 370];
    let gear = 1;
    for (let g = 1; g <= 8; g++) {
      if (speedKmh >= gearSpeeds[g - 1]) gear = g;
    }
    const gMin = gearSpeeds[gear - 1];
    const gMax = gearSpeeds[gear];
    const gearFrac = Math.max(0, Math.min(1, (speedKmh - gMin) / (gMax - gMin + 0.1)));

    /* Simulated F1 RPM range from 5000 idle up to 13800 near limiter */
    const rpm = 5000 + gearFrac * 8800;
    const baseFreq = Math.max(70, Math.min(720, (rpm / 60) * 3));

    const now = this.ctx.currentTime;
    this.oscSaw.frequency.setTargetAtTime(baseFreq, now, 0.03);
    this.oscSub.frequency.setTargetAtTime(baseFreq * 0.5, now, 0.03);
    this.bpFilter.frequency.setTargetAtTime(Math.min(2200, baseFreq * 2.6), now, 0.04);

    /* Turbo spool pitch and volume tracking throttle input */
    const turboFreq = 2200 + gearFrac * 1700;
    this.oscTurbo.frequency.setTargetAtTime(turboFreq, now, 0.04);
    this.turboGain.gain.setTargetAtTime(thr * 0.06, now, 0.05);

    /* Volume calculation with attenuation on engine overrun under heavy braking */
    let targetVol = (0.07 + thr * 0.15 + (speedKmh / 370) * 0.07) * volumeMultiplier;
    if (brk > 0.1) targetVol *= 0.85; /* Engine overrun and braking backpressure */
    this.gain.gain.setTargetAtTime(targetVol, now, 0.04);
  }

  stopOrIdle() {
    if (!this.ctx) return;
    const now = this.ctx.currentTime;
    this.gain.gain.setTargetAtTime(0.0, now, 0.1);
    this.turboGain.gain.setTargetAtTime(0.0, now, 0.1);
  }
}

class CrowdAmbience {
  constructor(ctx, dest) {
    this.ctx = ctx;
    this.dest = dest;
    this.gain = ctx.createGain();
    this.gain.gain.value = 0.0; /* I keep crowd gain at zero initially so it stays quiet during the hero intro */
    this.gain.connect(dest);

    this.hasExternalAudio = false;
    this.externalSource = null;
    this.audioBuffer = null;

    /* 1. Procedural pink noise with resonant grandstand formant filter */
    this.setupProceduralCrowd();

    /* 2. Automatically load custom audio if placed in renderer/assets/crowd_noise.mp3 */
    this.tryLoadAudioFile();
  }

  setupProceduralCrowd() {
    const ctx = this.ctx;
    const bufLen = ctx.sampleRate * 4;
    const buffer = ctx.createBuffer(1, bufLen, ctx.sampleRate);
    const data = buffer.getChannelData(0);
    let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
    for (let i = 0; i < bufLen; i++) {
      const white = Math.random() * 2 - 1;
      b0 = 0.99886 * b0 + white * 0.0555179;
      b1 = 0.99332 * b1 + white * 0.0750759;
      b2 = 0.96900 * b2 + white * 0.1538520;
      b3 = 0.86650 * b3 + white * 0.3104856;
      b4 = 0.55000 * b4 + white * 0.5329522;
      b5 = -0.7616 * b5 - white * 0.0168980;
      data[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362) * 0.08;
      b6 = white * 0.115926;
    }

    this.procSrc = ctx.createBufferSource();
    this.procSrc.buffer = buffer;
    this.procSrc.loop = true;

    this.filter = ctx.createBiquadFilter();
    this.filter.type = 'bandpass';
    this.filter.frequency.value = 420;
    this.filter.Q.value = 1.8;

    this.procGain = ctx.createGain();
    this.procGain.gain.value = 1.0;

    this.procSrc.connect(this.filter);
    this.filter.connect(this.procGain);
    this.procGain.connect(this.gain);

    this.procSrc.start();
  }

  async tryLoadAudioFile() {
    const candidates = ['assets/crowd_noise.mp3', 'assets/crowd_noise.wav'];
    for (const path of candidates) {
      try {
        const res = await fetch(path);
        if (!res.ok) continue;
        const arrayBuf = await res.arrayBuffer();
        const audioBuf = await new Promise((resolve, reject) => {
          this.ctx.decodeAudioData(arrayBuf, resolve, reject);
        });

        this.audioBuffer = audioBuf;
        this.hasExternalAudio = true;
        this.startExternalSource();

        if (this.procGain) {
          this.procGain.gain.setTargetAtTime(0, this.ctx.currentTime, 0.4);
        }
        console.log(`[F1 Audio] Loaded custom crowd audio from: ${path} (Audible gain active)`);
        break;
      } catch (err) {
        console.warn(`[F1 Audio] Could not load ${path}:`, err);
      }
    }
  }

  startExternalSource() {
    if (!this.audioBuffer || !this.ctx) return;
    if (this.externalSource) {
      try { this.externalSource.stop(); } catch (e) { }
    }
    this.externalSource = this.ctx.createBufferSource();
    this.externalSource.buffer = this.audioBuffer;
    this.externalSource.loop = true;
    this.externalSource.connect(this.gain);
    try {
      this.externalSource.start();
    } catch (e) {
      console.warn('[F1 Audio] externalSource.start failed:', e);
    }
  }

  ensurePlaying() {
    if (this.hasExternalAudio && !this.externalSource && this.audioBuffer) {
      this.startExternalSource();
    }
  }

  setExcitement(level) {
    const now = this.ctx.currentTime;
    /* Deliver rich crowd ambience when custom audio is present */
    const targetGain = this.hasExternalAudio
      ? 0.48 + level * 0.38 /* Baseline volume rising during high speed action */
      : 0.15 + level * 0.25; /* Procedural fallback volume */
    this.gain.gain.setTargetAtTime(targetGain, now, 0.1);
    if (this.filter && !this.hasExternalAudio) {
      const targetFreq = 420 + level * 350;
      this.filter.frequency.setTargetAtTime(targetFreq, now, 0.15);
    }
  }
}

class BroadcastMusic {
  constructor(ctx, dest) {
    this.ctx = ctx;
    this.dest = dest;
    this.gain = ctx.createGain();
    this.gain.gain.value = 0.0; /* I initialize this to zero so it will not play during hero mode */
    this.gain.connect(dest);

    this.hasExternalAudio = false;
    this.externalSource = null;

    /* 1. Procedural synth pad chord progression */
    this.setupProceduralMusic();

    /* 2. Automatically load custom audio if present in renderer/assets/ */
    this.tryLoadAudioFile();
  }

  setupProceduralMusic() {
    const ctx = this.ctx;
    this.procGain = ctx.createGain();
    this.procGain.gain.value = 1.0;
    this.procGain.connect(this.gain);

    this.lpFilter = ctx.createBiquadFilter();
    this.lpFilter.type = 'lowpass';
    this.lpFilter.frequency.value = 600;

    this.osc1 = ctx.createOscillator();
    this.osc1.type = 'triangle';
    this.osc1.frequency.value = 73.42; /* D2 note */

    this.osc2 = ctx.createOscillator();
    this.osc2.type = 'sine';
    this.osc2.frequency.value = 146.83; /* D3 note */

    this.osc3 = ctx.createOscillator();
    this.osc3.type = 'triangle';
    this.osc3.frequency.value = 220.00; /* A3 note */

    this.osc1.connect(this.lpFilter);
    this.osc2.connect(this.lpFilter);
    this.osc3.connect(this.lpFilter);
    this.lpFilter.connect(this.procGain);

    this.osc1.start();
    this.osc2.start();
    this.osc3.start();

    /* Subtle ambient chord cycle: D minor, B flat, F major, C major */
    this.chords = [
      [73.42, 146.83, 220.0],
      [58.27, 116.54, 174.61],
      [87.31, 174.61, 261.63],
      [65.41, 130.81, 196.0],
    ];
    this.step = 0;
    this.nextChange = 0;
  }

  async tryLoadAudioFile() {
    const candidates = ['assets/bg_sound.mp3', 'assets/bg_sound.wav'];
    for (const path of candidates) {
      try {
        const res = await fetch(path);
        if (!res.ok) continue;
        const arrayBuf = await res.arrayBuffer();
        const audioBuf = await new Promise((resolve, reject) => {
          this.ctx.decodeAudioData(arrayBuf, resolve, reject);
        });

        this.externalSource = this.ctx.createBufferSource();
        this.externalSource.buffer = audioBuf;
        this.externalSource.loop = true;
        this.externalSource.connect(this.gain);
        this.externalSource.start();

        if (this.procGain) {
          this.procGain.gain.setTargetAtTime(0, this.ctx.currentTime, 0.4);
        }
        this.hasExternalAudio = true;
        console.log(`[F1 Audio] Loaded custom background sound from: ${path}`);
        break;
      } catch (err) {
        /* Fallback smoothly to procedural synthesizer */
      }
    }
  }

  update() {
    if (this.hasExternalAudio) return;
    const now = this.ctx.currentTime;
    if (now >= this.nextChange) {
      this.step = (this.step + 1) % this.chords.length;
      const c = this.chords[this.step];
      this.osc1.frequency.setTargetAtTime(c[0], now, 0.9);
      this.osc2.frequency.setTargetAtTime(c[1], now, 0.9);
      this.osc3.frequency.setTargetAtTime(c[2], now, 0.9);
      this.nextChange = now + 4.8;
    }
  }
}

/*
 * Dedicated Hero Section BGM Player
 * Loads renderer/assets/hero_bgm.mp3 or hero_bgm.wav.
 * If file is not yet present, falls back gracefully to my procedural broadcast intro synthesizer.
 */
class HeroBGMPlayer {
  constructor(ctx, dest) {
    this.ctx = ctx;
    this.dest = dest;
    this.gain = ctx.createGain();
    this.gain.gain.value = 0.0; /* Starts silent, fades in on hero show */
    this.gain.connect(dest);

    this.hasExternalAudio = false;
    this.externalSource = null;
    this.audioBuffer = null;
    this.isPlaying = false;

    /* 1. Procedural broadcast intro theme fallback using ambient atmospheric arpeggio */
    this.setupProceduralTheme();

    /* 2. Load custom user file from renderer/assets/hero_bgm.mp3 or wav */
    this.tryLoadAudioFile();
  }

  setupProceduralTheme() {
    const ctx = this.ctx;
    this.procGain = ctx.createGain();
    this.procGain.gain.value = 0.0;
    this.procGain.connect(this.gain);

    this.lpFilter = ctx.createBiquadFilter();
    this.lpFilter.type = 'lowpass';
    this.lpFilter.frequency.value = 950;

    this.osc1 = ctx.createOscillator();
    this.osc1.type = 'sawtooth';
    this.osc1.frequency.value = 110.0; /* A2 note */

    this.osc2 = ctx.createOscillator();
    this.osc2.type = 'sine';
    this.osc2.frequency.value = 220.0; // A3

    this.osc3 = ctx.createOscillator();
    this.osc3.type = 'triangle';
    this.osc3.frequency.value = 329.63; /* E4 note */

    this.osc1.connect(this.lpFilter);
    this.osc2.connect(this.lpFilter);
    this.osc3.connect(this.lpFilter);
    this.lpFilter.connect(this.procGain);

    this.osc1.start();
    this.osc2.start();
    this.osc3.start();

    /* Hero intro cinematic chords: A minor, F major, C major, G major */
    this.chords = [
      [110.0, 220.0, 329.63],   /* A minor */
      [87.31, 174.61, 261.63],  /* F major */
      [130.81, 261.63, 329.63], /* C major */
      [98.0, 196.0, 293.66],    /* G major */
    ];
    this.step = 0;
    this.nextChange = 0;
  }

  async tryLoadAudioFile() {
    const candidates = ['assets/hero_bgm.mp3', 'assets/hero_bgm.wav'];
    for (const path of candidates) {
      try {
        const res = await fetch(path);
        if (!res.ok) continue;
        const arrayBuf = await res.arrayBuffer();
        const audioBuf = await new Promise((resolve, reject) => {
          this.ctx.decodeAudioData(arrayBuf, resolve, reject);
        });

        this.audioBuffer = audioBuf;
        this.hasExternalAudio = true;
        console.log(`[F1 Hero Audio] Loaded custom hero BGM from: ${path}`);
        if (this.isPlaying) {
          this._startBufferPlayback();
        }
        break;
      } catch (err) {
        /* Fallback to procedural synth */
      }
    }
  }

  _startBufferPlayback() {
    if (!this.audioBuffer || !this.ctx) return;
    if (this.externalSource) {
      try { this.externalSource.stop(); } catch (e) { }
    }
    this.externalSource = this.ctx.createBufferSource();
    this.externalSource.buffer = this.audioBuffer;
    this.externalSource.loop = true;
    this.externalSource.connect(this.gain);
    try {
      this.externalSource.start();
    } catch (e) { }

    if (this.procGain) {
      this.procGain.gain.setTargetAtTime(0, this.ctx.currentTime, 0.2);
    }
  }

  play() {
    this.isPlaying = true;
    const now = this.ctx.currentTime;
    this.gain.gain.cancelScheduledValues(now);
    this.gain.gain.setValueAtTime(this.gain.gain.value, now);
    /* Smooth cinematic fade in */
    this.gain.gain.setTargetAtTime(this.hasExternalAudio ? 0.65 : 0.35, now, 0.8);

    if (this.hasExternalAudio) {
      if (!this.externalSource && this.audioBuffer) {
        this._startBufferPlayback();
      }
    } else if (this.procGain) {
      this.procGain.gain.setTargetAtTime(1.0, now, 0.8);
    }
  }

  stop(fadeOutDuration = 1.0) {
    this.isPlaying = false;
    const now = this.ctx.currentTime;
    this.gain.gain.cancelScheduledValues(now);
    this.gain.gain.setValueAtTime(this.gain.gain.value, now);
    this.gain.gain.setTargetAtTime(0.0, now, fadeOutDuration);
  }

  update() {
    if (!this.isPlaying || this.hasExternalAudio) return;
    const now = this.ctx.currentTime;
    if (now >= this.nextChange) {
      this.step = (this.step + 1) % this.chords.length;
      const c = this.chords[this.step];
      this.osc1.frequency.setTargetAtTime(c[0], now, 0.8);
      this.osc2.frequency.setTargetAtTime(c[1], now, 0.8);
      this.osc3.frequency.setTargetAtTime(c[2], now, 0.8);
      this.nextChange = now + 4.5;
    }
  }
}

class F1SoundSystem {
  constructor() {
    this.ctx = null;
    this.masterGain = null;
    this.engineA = null;
    this.engineB = null;
    this.crowd = null;
    this.music = null;
    this.heroBgm = null;
    this.isMuted = false;
    this.initialized = false;
  }

  init() {
    if (this.initialized) return;
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AudioCtx();
      this.masterGain = this.ctx.createGain();
      this.masterGain.gain.setValueAtTime(0.75, this.ctx.currentTime);
      this.masterGain.connect(this.ctx.destination);

      this.engineA = new F1EngineSound(this.ctx, this.masterGain, { pan: -0.28 });
      this.engineB = new F1EngineSound(this.ctx, this.masterGain, { pan: 0.28 });
      this.crowd = new CrowdAmbience(this.ctx, this.masterGain);
      this.music = new BroadcastMusic(this.ctx, this.masterGain);
      this.heroBgm = new HeroBGMPlayer(this.ctx, this.masterGain);

      this.initialized = true;
    } catch (e) {
      console.warn('Web Audio init skipped:', e);
    }
  }

  ensureResumed() {
    if (!this.initialized) this.init();
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
    this.crowd?.ensurePlaying();
  }

  startHeroBGM() {
    this.ensureResumed();
    /* When starting the hero intro, I immediately silence car engines, crowd audio, and in race music */
    this.engineA?.stopOrIdle();
    this.engineB?.stopOrIdle();
    if (this.crowd && this.crowd.gain) {
      this.crowd.gain.gain.setValueAtTime(0.0, this.ctx.currentTime);
    }
    if (this.music && this.music.gain) {
      this.music.gain.gain.setValueAtTime(0.0, this.ctx.currentTime);
    }
    this.heroBgm?.play();
  }

  stopHeroBGM(fade = 1.0) {
    this.heroBgm?.stop(fade);
  }

  setMuted(muted) {
    this.isMuted = muted;
    if (this.masterGain && this.ctx) {
      this.masterGain.gain.setTargetAtTime(muted ? 0.0 : 0.75, this.ctx.currentTime, 0.05);
    }
  }

  toggleMute() {
    this.setMuted(!this.isMuted);
    return !this.isMuted;
  }

  /*
   * My sound update loop.
   * Crucially, when hero mode is active, I keep only the hero bgm playing and completely mute
   * car engine noises so the intro screen remains clean and focused on music.
   * Car noise is unleashed only during the main race.
   */
  update(sA, sB, isPlaying, cameraMode, isHeroMode = false) {
    if (!this.initialized || !this.ctx || this.ctx.state !== 'running') return;

    if (isHeroMode) {
      /* In hero mode I strictly update only the hero bgm and silence car noise and grandstands */
      this.heroBgm?.update();
      this.engineA?.stopOrIdle();
      this.engineB?.stopOrIdle();
      if (this.crowd && this.crowd.gain) {
        this.crowd.gain.gain.setTargetAtTime(0.0, this.ctx.currentTime, 0.05);
      }
      if (this.music && this.music.gain) {
        this.music.gain.gain.setTargetAtTime(0.0, this.ctx.currentTime, 0.05);
      }
      return;
    }

    if (!isPlaying) {
      this.engineA?.stopOrIdle();
      this.engineB?.stopOrIdle();
      this.crowd?.setExcitement(0.12);
      return;
    }

    /* In the main race, I update both car engines with volume priority given to the focused camera */
    const multA = cameraMode === 'carA' || cameraMode === 'cockpit' ? 1.4 : 1.0;
    const multB = cameraMode === 'carB' ? 1.4 : 1.0;
    this.engineA?.update(sA, multA);
    this.engineB?.update(sB, multB);

    /* I link crowd excitement to peak velocity so cheering rises naturally on high speed straights */
    const maxSpeed = Math.max(sA.speed, sB.speed);
    const speedKmh = maxSpeed * 3.6;
    const excitement = Math.max(0, Math.min(1, (speedKmh - 160) / 140));
    this.crowd?.setExcitement(excitement);

    this.music?.update();
  }
}

const soundSystem = new F1SoundSystem();

/*
 * Phase 2: 3D Circuit Geometry and Sunlit Environment
 *
 * Here I construct the procedural racetrack, asphalt surface, FIA curbs, safety barriers,
 * distance brake countdown markers, start line gantry, stadium lighting, and spectator grandstands.
 */

/*
 * I generate a procedural asphalt grain bump texture using an offscreen HTML5 canvas.
 * My approach generates subtle per pixel luminosity variations that simulate real aggregate bitumen stone,
 * providing realistic specular highlights under the afternoon sun without needing heavy texture downloads.
 */
function getAsphaltTexture() {
  if (getAsphaltTexture._cached) return getAsphaltTexture._cached;
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, 256, 256);
  const img = ctx.getImageData(0, 0, 256, 256);
  for (let i = 0; i < img.data.length; i += 4) {
    const n = (Math.random() - 0.5) * 55;
    const v = Math.max(0, Math.min(255, 128 + n));
    img.data[i] = v;
    img.data[i + 1] = v;
    img.data[i + 2] = v;
  }
  ctx.putImageData(img, 0, 0);
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(40, 40);
  getAsphaltTexture._cached = tex;
  return tex;
}

/*
 * Here I construct a continuous 3D ribbon mesh connecting two boundary polylines.
 * I generate vertices along both inner and outer edges, wire them into triangles with consistent normals,
 * and calculate UV coordinates mapped to track length so asphalt textures tile seamlessly.
 */
function buildRibbonMesh(polyA, polyB, material, y_offset = 0) {
  const n = polyA.length;
  const positions = [];
  const indices = [];

  for (let i = 0; i < n; i++) {
    positions.push(polyA[i][0], y_offset, -polyA[i][1]);
    positions.push(polyB[i][0], y_offset, -polyB[i][1]);
  }

  for (let i = 0; i < n; i++) {
    const next = (i + 1) % n;
    const a = i * 2;
    const b = i * 2 + 1;
    const c = next * 2;
    const d = next * 2 + 1;
    indices.push(a, c, b);
    indices.push(b, c, d);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();

  const mesh = new THREE.Mesh(geometry, material);
  mesh.receiveShadow = true;
  return mesh;
}

/*
 * I construct vertical Armco crash barriers along the outer circuit boundaries.
 * I generate double sided vertical quads and top them with high visibility safety rails
 * colored in FIA racing blue, creating a clear visual boundary for track limits.
 */
function buildBarrierWall(poly, height, baseColor = 0x161c28) {
  const group = new THREE.Group();
  const n = poly.length;
  const positions = [];
  const indices = [];

  for (let i = 0; i < n; i++) {
    positions.push(poly[i][0], 0, -poly[i][1]);
    positions.push(poly[i][0], height, -poly[i][1]);
  }

  for (let i = 0; i < n; i++) {
    const next = (i + 1) % n;
    const bl = i * 2;
    const tl = i * 2 + 1;
    const br = next * 2;
    const tr = next * 2 + 1;
    indices.push(bl, br, tl);
    indices.push(tl, br, tr);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();

  const mat = new THREE.MeshStandardMaterial({
    color: baseColor,
    roughness: 0.65,
    metalness: 0.25,
    side: THREE.DoubleSide,
  });

  const wallMesh = new THREE.Mesh(geometry, mat);
  wallMesh.receiveShadow = true;
  group.add(wallMesh);

  /* Top illuminated electric cyan safety rail running along the barrier */
  const railPoints = poly.map(p => new THREE.Vector3(p[0], height + 0.05, -p[1]));
  railPoints.push(railPoints[0].clone());
  const railGeo = new THREE.BufferGeometry().setFromPoints(railPoints);
  const railLine = new THREE.Line(railGeo, new THREE.LineBasicMaterial({ color: 0x00f0ff, linewidth: 3 }));
  group.add(railLine);

  return group;
}

/*
 * Here I generate crisp white track boundary lines.
 * I elevate them slightly above the asphalt surface by 35 millimeters to prevent z fighting,
 * creating sharp edges defining the legal racing surface.
 */
function buildEdgeLine(poly, color = 0xffffff, opacity = 0.95) {
  const points = poly.map(p => new THREE.Vector3(p[0], 0.035, -p[1]));
  points.push(points[0].clone());
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({
    color: color,
    opacity: opacity,
    transparent: true,
  });
  return new THREE.Line(geometry, material);
}

/*
 * I generate 3D rumble curbs placed along corner apexes and exit zones.
 * My approach identifies sharp curvature along the track boundaries, extrudes beveled curb geometry,
 * and batches all segments into two high performance merged meshes to minimize draw calls.
 */
function buildCurbs(track) {
  const group = new THREE.Group();
  const curvature = track.curvature;
  const left = track.left_boundary;
  const right = track.right_boundary;
  const n = left.length;
  const CURB_WIDTH = 0.85;
  const CURB_THRESHOLD = 0.012;

  const redMat = new THREE.MeshStandardMaterial({ color: 0xe10600, roughness: 0.35, metalness: 0.1 });
  const whiteMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.35, metalness: 0.1 });

  const redPositions = [];
  const redIndices = [];
  let redIdx = 0;

  const whitePositions = [];
  const whiteIndices = [];
  let whiteIdx = 0;

  for (let i = 0; i < n; i++) {
    if (Math.abs(curvature[i]) < CURB_THRESHOLD) continue;

    const next = (i + 1) % n;
    const isLeftCurb = curvature[i] < 0;
    const boundary = isLeftCurb ? left : right;

    const bCurr = boundary[i];
    const bNext = boundary[next];
    const dx = bNext[0] - bCurr[0];
    const dy = bNext[1] - bCurr[1];
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len < 0.001) continue;

    const nx = isLeftCurb ? dy / len : -dy / len;
    const ny = isLeftCurb ? -dx / len : dx / len;

    const isRed = Math.floor(i / 2) % 2 === 0;

    const p0 = [bCurr[0], bCurr[1]];
    const p1 = [bNext[0], bNext[1]];
    const p2 = [bNext[0] + nx * CURB_WIDTH, bNext[1] + ny * CURB_WIDTH];
    const p3 = [bCurr[0] + nx * CURB_WIDTH, bCurr[1] + ny * CURB_WIDTH];

    if (isRed) {
      redPositions.push(
        p0[0], 0.02, -p0[1],
        p1[0], 0.02, -p1[1],
        p2[0], 0.065, -p2[1],
        p3[0], 0.065, -p3[1]
      );
      redIndices.push(redIdx, redIdx + 1, redIdx + 2, redIdx, redIdx + 2, redIdx + 3);
      redIdx += 4;
    } else {
      whitePositions.push(
        p0[0], 0.02, -p0[1],
        p1[0], 0.02, -p1[1],
        p2[0], 0.065, -p2[1],
        p3[0], 0.065, -p3[1]
      );
      whiteIndices.push(whiteIdx, whiteIdx + 1, whiteIdx + 2, whiteIdx, whiteIdx + 2, whiteIdx + 3);
      whiteIdx += 4;
    }
  }

  /* I batch all red and white curb segments into two merged geometries to minimize draw calls */
  if (redPositions.length > 0) {
    const redGeo = new THREE.BufferGeometry();
    redGeo.setAttribute('position', new THREE.Float32BufferAttribute(redPositions, 3));
    redGeo.setIndex(redIndices);
    redGeo.computeVertexNormals();
    const redMesh = new THREE.Mesh(redGeo, redMat);
    redMesh.receiveShadow = true;
    group.add(redMesh);
  }

  if (whitePositions.length > 0) {
    const whiteGeo = new THREE.BufferGeometry();
    whiteGeo.setAttribute('position', new THREE.Float32BufferAttribute(whitePositions, 3));
    whiteGeo.setIndex(whiteIndices);
    whiteGeo.computeVertexNormals();
    const whiteMesh = new THREE.Mesh(whiteGeo, whiteMat);
    whiteMesh.receiveShadow = true;
    group.add(whiteMesh);
  }

  return group;
}

/*
 * Distance brake marker boards positioned approaching heavy braking zones.
 * I mount these on the outer barrier walls using instanced geometry to ensure peak rendering speed.
 */
function buildBrakeMarkers(track) {
  const group = new THREE.Group();
  const n = track.centerline.length;
  const boardGeo = new THREE.BoxGeometry(0.85, 1.2, 0.08);
  const boardMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.25 });
  const transforms = [];

  for (let i = 0; i < n; i += 22) {
    if (Math.abs(track.curvature[i]) > 0.015) {
      const isLeft = track.curvature[i] < 0;
      const b = isLeft ? track.barrier_left : track.barrier_right;
      for (let step = 1; step <= 3; step++) {
        const targetIdx = (i - step * 5 + n) % n;
        const pt = b[targetIdx];
        const m = new THREE.Matrix4();
        m.setPosition(pt[0], 1.25, -pt[1]);
        transforms.push(m);
      }
    }
  }

  if (transforms.length > 0) {
    const instMesh = new THREE.InstancedMesh(boardGeo, boardMat, transforms.length);
    for (let i = 0; i < transforms.length; i++) {
      instMesh.setMatrixAt(i, transforms[i]);
    }
    instMesh.instanceMatrix.needsUpdate = true;
    group.add(instMesh);
  }

  return group;
}

function createGridSlotTexture(slotNum, isPole) {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 1024;
  const ctx = canvas.getContext('2d');

  ctx.clearRect(0, 0, 512, 1024);

  /* Outer FIA white boundary box */
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 18;
  ctx.strokeRect(24, 40, 464, 940);

  /* Front stop transverse bar marking front tire stopping point */
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(14, 40, 484, 46);

  /* Center yellow alignment line */
  ctx.fillStyle = '#facc15';
  ctx.fillRect(244, 90, 24, 880);

  /* Left and right front tire alignment chevrons */
  ctx.fillStyle = '#facc15';
  ctx.beginPath();
  ctx.moveTo(80, 160); ctx.lineTo(120, 110); ctx.lineTo(160, 160);
  ctx.lineTo(145, 160); ctx.lineTo(120, 130); ctx.lineTo(95, 160);
  ctx.closePath();
  ctx.fill();

  ctx.beginPath();
  ctx.moveTo(352, 160); ctx.lineTo(392, 110); ctx.lineTo(432, 160);
  ctx.lineTo(417, 160); ctx.lineTo(392, 130); ctx.lineTo(367, 160);
  ctx.closePath();
  ctx.fill();

  /* Painted position numeral box on the tarmac */
  ctx.fillStyle = isPole ? '#00f0ff' : '#ff5500';
  ctx.fillRect(176, 230, 160, 140);
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 8;
  ctx.strokeRect(176, 230, 160, 140);

  ctx.fillStyle = '#0f172a';
  ctx.font = '900 110px "Titillium Web", sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(String(slotNum), 256, 300);

  /* Label: POLE, P1, or P2 */
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 36px "Titillium Web", sans-serif';
  ctx.fillText(isPole ? 'POLE' : 'P2', 256, 410);

  const tex = new THREE.CanvasTexture(canvas);
  tex.anisotropy = 8;
  return tex;
}

/*
 * Start and finish checkered grid line and overhead gantry structure.
 * I place all gantry columns strictly outside the barrier perimeter for track safety compliance.
 */
function buildStartFinishStructure(track, trackCenter) {
  const group = new THREE.Group();
  const idx = 0;
  const left = track.left_boundary[idx];
  const right = track.right_boundary[idx];
  const bLeft = track.barrier_left[idx];
  const bRight = track.barrier_right[idx];

  const dx = right[0] - left[0];
  const dy = right[1] - left[1];
  const segments = 12;
  const lineWidth = 2.0;

  const next = 1;
  const tdx = track.centerline[next][0] - track.centerline[idx][0];
  const tdy = track.centerline[next][1] - track.centerline[idx][1];
  const tLen = Math.hypot(tdx, tdy) || 1;
  const fwdX = tdx / tLen;
  const fwdY = tdy / tLen;

  /* 1. Checkered Finish Line on Asphalt */
  for (let s = 0; s < segments; s++) {
    const t0 = s / segments;
    const t1 = (s + 1) / segments;
    const isWhite = s % 2 === 0;

    const lx0 = left[0] + dx * t0;
    const ly0 = left[1] + dy * t0;
    const lx1 = left[0] + dx * t1;
    const ly1 = left[1] + dy * t1;

    const positions = [
      lx0 - fwdX * lineWidth / 2, 0.035, -(ly0 - fwdY * lineWidth / 2),
      lx1 - fwdX * lineWidth / 2, 0.035, -(ly1 - fwdY * lineWidth / 2),
      lx1 + fwdX * lineWidth / 2, 0.035, -(ly1 + fwdY * lineWidth / 2),
      lx0 + fwdX * lineWidth / 2, 0.035, -(ly0 + fwdY * lineWidth / 2),
    ];

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.setIndex([0, 1, 2, 0, 2, 3]);
    geo.computeVertexNormals();

    const mat = new THREE.MeshStandardMaterial({
      color: isWhite ? 0xffffff : 0x1a1e24,
      roughness: 0.4,
      metalness: 0.1,
    });
    group.add(new THREE.Mesh(geo, mat));
  }

  /* 2. Official FIA F1 Staggered Grid Slots for Pole and Slot 2 */
  for (const [slotNum, sideOffset, isPole] of [[1, -1.35, true], [2, 1.35, false]]) {
    const staggerBack = isPole ? 3.0 : 12.0; /* P1 is positioned ahead, P2 is staggered 9 meters behind */
    const slotX = track.centerline[idx][0] - fwdX * staggerBack;
    const slotY = track.centerline[idx][1] - fwdY * staggerBack;
    const perpX = -fwdY * sideOffset;
    const perpY = fwdX * sideOffset;

    const boxGeo = new THREE.PlaneGeometry(2.5, 5.2);
    const boxTex = createGridSlotTexture(slotNum, isPole);
    const boxMat = new THREE.MeshStandardMaterial({
      map: boxTex,
      transparent: true,
      roughness: 0.35,
      metalness: 0.1,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });
    const boxMesh = new THREE.Mesh(boxGeo, boxMat);
    boxMesh.rotation.x = -Math.PI / 2;
    boxMesh.rotation.z = Math.atan2(fwdX, fwdY) + Math.PI;
    boxMesh.position.set(slotX + perpX, 0.038, -(slotY + perpY));
    group.add(boxMesh);
  }

  /* 3. Overhead Gantry Structure */
  const c0 = track.centerline[idx];
  const toLeftX = bLeft[0] - c0[0];
  const toLeftY = bLeft[1] - c0[1];
  const lenL = Math.hypot(toLeftX, toLeftY) || 1;
  const pillarLeftPos = new THREE.Vector3(bLeft[0] + (toLeftX / lenL) * 3.0, 0, -(bLeft[1] + (toLeftY / lenL) * 3.0));

  const toRightX = bRight[0] - c0[0];
  const toRightY = bRight[1] - c0[1];
  const lenR = Math.hypot(toRightX, toRightY) || 1;
  const pillarRightPos = new THREE.Vector3(bRight[0] + (toRightX / lenR) * 3.0, 0, -(bRight[1] + (toRightY / lenR) * 3.0));

  const gantryHeight = 9.5;
  const pillarMat = new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.8, roughness: 0.3 });
  const trussMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.7, roughness: 0.4 });

  const towerLeft = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.7, gantryHeight, 8), pillarMat);
  towerLeft.position.set(pillarLeftPos.x, gantryHeight / 2, pillarLeftPos.z);
  towerLeft.castShadow = true;
  group.add(towerLeft);

  const towerRight = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.7, gantryHeight, 8), pillarMat);
  towerRight.position.set(pillarRightPos.x, gantryHeight / 2, pillarRightPos.z);
  towerRight.castShadow = true;
  group.add(towerRight);

  const span = pillarLeftPos.distanceTo(pillarRightPos);
  const midPos = pillarLeftPos.clone().add(pillarRightPos).multiplyScalar(0.5);
  midPos.y = gantryHeight - 0.5;

  const crossBeam = new THREE.Mesh(new THREE.BoxGeometry(span, 1.2, 1.8), trussMat);
  crossBeam.position.copy(midPos);
  const bridgeAngle = Math.atan2(pillarRightPos.z - pillarLeftPos.z, pillarRightPos.x - pillarLeftPos.x);
  crossBeam.rotation.y = -bridgeAngle;
  crossBeam.castShadow = true;
  group.add(crossBeam);

  /* Overhead illuminated poster display board mounted across the gantry crossbeam */
  const posterTex = getPosterTexture();
  const gantrySignMat = new THREE.MeshStandardMaterial({
    map: posterTex,
    emissive: 0xffffff,
    emissiveMap: posterTex,
    emissiveIntensity: 0.65,
    roughness: 0.2,
    metalness: 0.1,
    side: THREE.DoubleSide,
  });
  const gantrySignGeo = new THREE.PlaneGeometry(Math.min(span * 0.72, 14.0), 1.6);
  const gantrySign = new THREE.Mesh(gantrySignGeo, gantrySignMat);
  gantrySign.position.set(midPos.x, midPos.y + 0.35, midPos.z);
  gantrySign.rotation.y = -bridgeAngle;
  group.add(gantrySign);

  /* 5 Red Starting Lights Mounted on Crossbeam High Above */
  const lightGeo = new THREE.SphereGeometry(0.28, 16, 16);
  const lightMat = new THREE.MeshBasicMaterial({ color: 0xff0022 });
  const lightHousingMat = new THREE.MeshStandardMaterial({ color: 0x0f172a });

  const housing = new THREE.Mesh(new THREE.BoxGeometry(6.5, 0.9, 0.6), lightHousingMat);
  housing.position.set(midPos.x, midPos.y - 0.7, midPos.z);
  housing.rotation.y = -bridgeAngle;
  group.add(housing);

  for (let l = -2; l <= 2; l++) {
    const light = new THREE.Mesh(lightGeo, lightMat);
    light.position.set(midPos.x + Math.cos(-bridgeAngle) * l * 1.2, midPos.y - 0.7, midPos.z - Math.sin(-bridgeAngle) * l * 1.2);
    group.add(light);
  }

  return group;
}

/*
 * I cache the official Synapse Grand Prix poster loaded from assets/poster.svg
 * and generate glowing trackside digital sponsor hoardings.
 */
let posterTexture = null;
function getPosterTexture() {
  if (!posterTexture) {
    const loader = new THREE.TextureLoader();
    posterTexture = loader.load('assets/poster.svg');
    posterTexture.colorSpace = THREE.SRGBColorSpace;
    posterTexture.anisotropy = 8;
  }
  return posterTexture;
}

/*
 * I cache the second official Synapse Grand Prix poster loaded from assets/poster2.svg
 */
let poster2Texture = null;
function getPoster2Texture() {
  if (!poster2Texture) {
    const loader = new THREE.TextureLoader();
    poster2Texture = loader.load('assets/poster2.svg');
    poster2Texture.colorSpace = THREE.SRGBColorSpace;
    poster2Texture.anisotropy = 8;
  }
  return poster2Texture;
}

let synapseBannerTexture = null;
function getSynapseBannerTexture() {
  if (!synapseBannerTexture) {
    const c = document.createElement('canvas');
    c.width = 1024;
    c.height = 256;
    const ctx = c.getContext('2d');

    /* Deep cyber gradient backdrop */
    const grad = ctx.createLinearGradient(0, 0, 1024, 256);
    grad.addColorStop(0, '#060a12');
    grad.addColorStop(0.5, '#0b1329');
    grad.addColorStop(1, '#060a12');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 1024, 256);

    /* Electric neon border lines */
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 10;
    ctx.strokeRect(8, 8, 1008, 240);

    /* Top red FIA accent bar */
    ctx.fillStyle = '#e10600';
    ctx.fillRect(8, 8, 1008, 26);

    /* Telemetry speed line graph in background */
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.25)';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(30, 200);
    ctx.lineTo(200, 160);
    ctx.lineTo(400, 190);
    ctx.lineTo(600, 120);
    ctx.lineTo(800, 170);
    ctx.lineTo(990, 110);
    ctx.stroke();

    /* Chevrons */
    ctx.fillStyle = '#facc15';
    for (let x = 60; x <= 160; x += 35) {
      ctx.beginPath();
      ctx.moveTo(x, 70); ctx.lineTo(x + 20, 130); ctx.lineTo(x, 190);
      ctx.lineTo(x + 12, 190); ctx.lineTo(x + 32, 130); ctx.lineTo(x + 12, 70);
      ctx.closePath();
      ctx.fill();
    }

    /* Main brand typography */
    ctx.fillStyle = '#ffffff';
    ctx.font = '900 68px "Titillium Web", sans-serif';
    ctx.fillText('SYNAPSE AI RACING', 240, 135);

    ctx.fillStyle = '#00f0ff';
    ctx.font = 'bold 28px "Titillium Web", sans-serif';
    ctx.fillText('FORMULA 1 AUTONOMOUS GRAND PRIX // LIVE', 244, 185);

    synapseBannerTexture = new THREE.CanvasTexture(c);
    synapseBannerTexture.colorSpace = THREE.SRGBColorSpace;
    synapseBannerTexture.anisotropy = 8;
  }
  return synapseBannerTexture;
}

let screenGridTextureCached = null;
function getScreenGridTexture() {
  if (screenGridTextureCached) return screenGridTextureCached;
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'rgba(255, 255, 255, 0.96)';
  ctx.fillRect(0, 0, 64, 64);
  /* Horizontal scanline dark bands */
  ctx.fillStyle = 'rgba(0, 0, 0, 0.38)';
  for (let y = 0; y < 64; y += 4) {
    ctx.fillRect(0, y, 64, 1.6);
  }
  /* Subtle vertical LED sub pixel divisions */
  ctx.fillStyle = 'rgba(0, 0, 0, 0.20)';
  for (let x = 0; x < 64; x += 8) {
    ctx.fillRect(x, 0, 1.2, 64);
  }
  screenGridTextureCached = new THREE.CanvasTexture(canvas);
  screenGridTextureCached.wrapS = THREE.RepeatWrapping;
  screenGridTextureCached.wrapT = THREE.RepeatWrapping;
  screenGridTextureCached.repeat.set(16, 4);
  return screenGridTextureCached;
}

let screenReflectionTextureCached = null;
function getScreenReflectionTexture() {
  if (screenReflectionTextureCached) return screenReflectionTextureCached;
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');
  /* Anisotropic specular streak on glossy asphalt surface */
  const grad = ctx.createLinearGradient(0, 0, 0, 256);
  grad.addColorStop(0, 'rgba(255, 255, 255, 0.95)');
  grad.addColorStop(0.08, 'rgba(255, 255, 255, 0.82)');
  grad.addColorStop(0.24, 'rgba(255, 255, 255, 0.58)');
  grad.addColorStop(0.50, 'rgba(255, 255, 255, 0.32)');
  grad.addColorStop(0.78, 'rgba(255, 255, 255, 0.12)');
  grad.addColorStop(1.0, 'rgba(255, 255, 255, 0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 256, 256);
  /* Lateral soft feathering matching width of screen */
  ctx.globalCompositeOperation = 'destination-in';
  const horizGrad = ctx.createLinearGradient(0, 0, 256, 0);
  horizGrad.addColorStop(0, 'rgba(0, 0, 0, 0)');
  horizGrad.addColorStop(0.12, 'rgba(0, 0, 0, 1)');
  horizGrad.addColorStop(0.88, 'rgba(0, 0, 0, 1)');
  horizGrad.addColorStop(1.0, 'rgba(0, 0, 0, 0)');
  ctx.fillStyle = horizGrad;
  ctx.fillRect(0, 0, 256, 256);
  screenReflectionTextureCached = new THREE.CanvasTexture(canvas);
  return screenReflectionTextureCached;
}



/*
 * I construct illuminated electronic barrier sponsor displays along left and right track barriers,
 * displaying the official Synapse Grand Prix posters and live Synapse AI Racing boards.
 * My approach batches screens, structural bezels, concrete wall reflections, and colored asphalt reflections
 * into hardware instanced meshes to guarantee maximum frame rates.
 */
function buildTracksidePosters(track, trackCenter) {
  const group = new THREE.Group();
  const n = track.centerline.length;
  const posterTex = getPosterTexture();
  const poster2Tex = getPoster2Texture();
  const synapseTex = getSynapseBannerTexture();

  const posterMat1 = new THREE.MeshStandardMaterial({
    map: posterTex,
    emissive: 0xffffff,
    emissiveMap: posterTex,
    emissiveIntensity: 1.55,
    roughness: 0.18,
    metalness: 0.05,
    side: THREE.DoubleSide,
  });

  const posterMat2 = new THREE.MeshStandardMaterial({
    map: poster2Tex,
    emissive: 0xffffff,
    emissiveMap: poster2Tex,
    emissiveIntensity: 1.55,
    roughness: 0.18,
    metalness: 0.05,
    side: THREE.DoubleSide,
  });

  const synapseMat = new THREE.MeshStandardMaterial({
    map: synapseTex,
    emissive: 0xffffff,
    emissiveMap: synapseTex,
    emissiveIntensity: 1.60,
    roughness: 0.18,
    metalness: 0.05,
    side: THREE.DoubleSide,
  });

  const neonCyan = new THREE.MeshBasicMaterial({ color: 0x00f0ff });
  const neonRed = new THREE.MeshBasicMaterial({ color: 0xff1133 });
  const neonGold = new THREE.MeshBasicMaterial({ color: 0xfacc15 });

  const sponsorConfigs = [
    { mat: posterMat1, neonMat: neonRed, colorHex: 0xff1133 },
    { mat: posterMat2, neonMat: neonGold, colorHex: 0xfacc15 },
    { mat: synapseMat, neonMat: neonCyan, colorHex: 0x00f0ff }
  ];

  const frameMat = new THREE.MeshStandardMaterial({ color: 0x070b12, roughness: 0.45, metalness: 0.90 });
  const mountBracketMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.35, metalness: 0.85 });

  const boardW = 6.0;
  const boardH = 1.15;
  const boardGeo = new THREE.PlaneGeometry(boardW, boardH);
  const backGeo = new THREE.BoxGeometry(boardW + 0.16, boardH + 0.12, 0.12);
  const bezelGeo = new THREE.BoxGeometry(boardW + 0.16, boardH + 0.16, 0.08);
  const bracketGeo = new THREE.BoxGeometry(0.12, boardH + 0.18, 0.22);

  const reflGeo = new THREE.PlaneGeometry(boardW + 1.4, 6.8);
  const wallReflGeo = new THREE.PlaneGeometry(boardW + 0.4, 0.62);
  const reflTex = getScreenReflectionTexture();

  const reflMaterials = sponsorConfigs.map(c => new THREE.MeshBasicMaterial({
    map: reflTex,
    color: c.colorHex,
    transparent: true,
    opacity: 0.82,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.DoubleSide,
    fog: false,
  }));

  const wallReflMaterials = sponsorConfigs.map(c => new THREE.MeshBasicMaterial({
    color: c.colorHex,
    transparent: true,
    opacity: 0.35,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.DoubleSide,
    fog: false,
  }));

  const barriers = [
    { poly: track.barrier_left, sideOffset: 0.12, isLeft: true },
    { poly: track.barrier_right, sideOffset: 0.12, isLeft: false }
  ];

  const step = Math.max(8, Math.floor(n / 16));
  const boardData = [];
  let boardIndex = 0;

  for (const bCfg of barriers) {
    const poly = bCfg.poly;
    for (let i = 0; i < n; i += step) {
      const bPt = poly[i];
      const cPt = track.centerline[i];

      const dirX = cPt[0] - bPt[0];
      const dirZ = -(cPt[1] - bPt[1]);
      const dist = Math.hypot(dirX, dirZ) || 1;
      const nx = dirX / dist;
      const nz = dirZ / dist;

      const posX = bPt[0] + nx * bCfg.sideOffset;
      const posZ = -bPt[1] + nz * bCfg.sideOffset;
      const posY = 0.68;
      const faceAngle = Math.atan2(nx, nz);

      const themeIdx = boardIndex % sponsorConfigs.length;
      boardIndex++;

      boardData.push({ posX, posY, posZ, faceAngle, themeIdx });
    }
  }

  const totalBoards = boardData.length;
  if (totalBoards === 0) return group;

  const maxPerTheme = Math.ceil(totalBoards / sponsorConfigs.length) + 8;
  const faceInstances = sponsorConfigs.map(c => new THREE.InstancedMesh(boardGeo, c.mat, maxPerTheme));
  const bezelInstances = sponsorConfigs.map(c => new THREE.InstancedMesh(bezelGeo, c.neonMat, maxPerTheme));
  const wallReflInstances = wallReflMaterials.map(m => new THREE.InstancedMesh(wallReflGeo, m, maxPerTheme));
  const reflInstances = reflMaterials.map(m => new THREE.InstancedMesh(reflGeo, m, maxPerTheme));
  const housingInstance = new THREE.InstancedMesh(backGeo, frameMat, totalBoards);
  const bracketInstance = new THREE.InstancedMesh(bracketGeo, mountBracketMat, totalBoards * 2);

  const themeCounts = [0, 0, 0];
  let housingCount = 0;
  let bracketCount = 0;

  const parentDummy = new THREE.Object3D();
  const subDummy = new THREE.Object3D();
  const combMat = new THREE.Matrix4();

  for (let b = 0; b < totalBoards; b++) {
    const d = boardData[b];
    const t = d.themeIdx;
    const tCount = themeCounts[t]++;

    parentDummy.position.set(d.posX, d.posY, d.posZ);
    parentDummy.rotation.set(0, d.faceAngle, 0);
    parentDummy.updateMatrix();

    /* Screen face */
    subDummy.position.set(0, 0, 0.065);
    subDummy.rotation.set(0, 0, 0);
    subDummy.updateMatrix();
    combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);
    faceInstances[t].setMatrixAt(tCount, combMat);

    /* Protective housing */
    subDummy.position.set(0, 0, 0);
    subDummy.updateMatrix();
    combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);
    housingInstance.setMatrixAt(housingCount++, combMat);

    /* Neon bezel frame */
    subDummy.position.set(0, 0, 0.01);
    subDummy.updateMatrix();
    combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);
    bezelInstances[t].setMatrixAt(tCount, combMat);

    /* Left and right mounting brackets */
    subDummy.position.set(-boardW * 0.48, 0, -0.04);
    subDummy.updateMatrix();
    combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);
    bracketInstance.setMatrixAt(bracketCount++, combMat);

    subDummy.position.set(boardW * 0.48, 0, -0.04);
    subDummy.updateMatrix();
    combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);
    bracketInstance.setMatrixAt(bracketCount++, combMat);

    /* Concrete barrier wall reflection */
    subDummy.position.set(0, -boardH * 0.5 - 0.28, 0.055);
    subDummy.updateMatrix();
    combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);
    wallReflInstances[t].setMatrixAt(tCount, combMat);

    /* Ground subtle puddle reflection */
    subDummy.position.set(0, -d.posY + 0.03, 1.8);
    subDummy.rotation.set(-Math.PI / 2, 0, 0);
    subDummy.scale.set(1, 1, 1);
    subDummy.updateMatrix();
    combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);
    reflInstances[t].setMatrixAt(tCount, combMat);
  }

  for (let t = 0; t < sponsorConfigs.length; t++) {
    faceInstances[t].count = themeCounts[t];
    faceInstances[t].instanceMatrix.needsUpdate = true;
    group.add(faceInstances[t]);

    bezelInstances[t].count = themeCounts[t];
    bezelInstances[t].instanceMatrix.needsUpdate = true;
    group.add(bezelInstances[t]);

    wallReflInstances[t].count = themeCounts[t];
    wallReflInstances[t].instanceMatrix.needsUpdate = true;
    group.add(wallReflInstances[t]);

    reflInstances[t].count = themeCounts[t];
    reflInstances[t].instanceMatrix.needsUpdate = true;
    group.add(reflInstances[t]);
  }

  housingInstance.count = housingCount;
  housingInstance.instanceMatrix.needsUpdate = true;
  group.add(housingInstance);

  bracketInstance.count = bracketCount;
  bracketInstance.instanceMatrix.needsUpdate = true;
  group.add(bracketInstance);

  return group;
}

/*
 * I erect towering stadium floodlight pylons surrounding the circuit perimeter.
 * Each pylon features a steel structural mast, an angled floodlight gantry head with bright emissive halogen LED banks,
 * and genuine directional spotlights illuminating the asphalt, grandstands, and cars.
 * My approach batches all pylon columns, platforms, heads, and floodlight banks into instanced meshes.
 */
function buildStadiumLights(track, trackCenter) {
  const group = new THREE.Group();
  const n = track.centerline.length;
  const step = Math.max(8, Math.floor(n / 14));

  const poleMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.85, roughness: 0.3 });
  const platformMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.7, roughness: 0.4 });
  const lampMat = new THREE.MeshBasicMaterial({ color: 0xfffdf5 });
  const lampHousingMat = new THREE.MeshStandardMaterial({ color: 0x0a0f1d, roughness: 0.5 });

  const poleGeo = new THREE.CylinderGeometry(0.45, 0.9, 26, 10);
  const platformGeo = new THREE.CylinderGeometry(2.4, 2.2, 0.4, 8);
  const headFrameGeo = new THREE.BoxGeometry(6.2, 2.2, 0.8);
  const bulbGeo = new THREE.BoxGeometry(1.2, 0.75, 0.2);

  const towerData = [];
  for (let i = 0; i < n; i += step) {
    const cp = track.centerline[i];
    const bL = track.barrier_left[i];
    const bR = track.barrier_right[i];

    const distL = Math.hypot(bL[0] - trackCenter.x, bL[1] - trackCenter.y);
    const distR = Math.hypot(bR[0] - trackCenter.x, bR[1] - trackCenter.y);
    const outerBarrier = distL > distR ? bL : bR;

    const normX = outerBarrier[0] - cp[0];
    const normY = outerBarrier[1] - cp[1];
    const normLen = Math.hypot(normX, normY) || 1;

    const poleX = outerBarrier[0] + (normX / normLen) * 22;
    const poleZ = -(outerBarrier[1] + (normY / normLen) * 22);
    const poleHeight = 26;

    let minTrackDist = Infinity;
    for (let k = 0; k < n; k += 4) {
      const d = Math.hypot(poleX - track.centerline[k][0], -poleZ - track.centerline[k][1]);
      if (d < minTrackDist) minTrackDist = d;
    }
    if (minTrackDist < 20.0) continue;

    const trackPtX = cp[0];
    const trackPtZ = -cp[1];
    const lookAngle = Math.atan2(trackPtX - poleX, trackPtZ - poleZ);

    towerData.push({ poleX, poleZ, poleHeight, lookAngle, trackPtX, trackPtZ });
  }

  const numTowers = towerData.length;
  if (numTowers === 0) return group;

  const poleInst = new THREE.InstancedMesh(poleGeo, poleMat, numTowers);
  const platformInst = new THREE.InstancedMesh(platformGeo, platformMat, numTowers);
  const headFrameInst = new THREE.InstancedMesh(headFrameGeo, lampHousingMat, numTowers);
  const bulbInst = new THREE.InstancedMesh(bulbGeo, lampMat, numTowers * 8);

  const parentDummy = new THREE.Object3D();
  const subDummy = new THREE.Object3D();
  const combMat = new THREE.Matrix4();
  let bulbCount = 0;
  let spotLightCount = 0;

  for (let t = 0; t < numTowers; t++) {
    const d = towerData[t];

    /* Structural pole column */
    subDummy.position.set(d.poleX, d.poleHeight / 2, d.poleZ);
    subDummy.rotation.set(0, 0, 0);
    subDummy.updateMatrix();
    poleInst.setMatrixAt(t, subDummy.matrix);

    /* Maintenance platform */
    subDummy.position.set(d.poleX, d.poleHeight - 1.2, d.poleZ);
    subDummy.updateMatrix();
    platformInst.setMatrixAt(t, subDummy.matrix);

    /* Floodlight head */
    parentDummy.position.set(d.poleX, d.poleHeight, d.poleZ);
    parentDummy.rotation.set(0, d.lookAngle, 0);
    parentDummy.updateMatrix();

    subDummy.position.set(0, 0, 0);
    subDummy.rotation.set(0.4, 0, 0);
    subDummy.updateMatrix();
    combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);
    headFrameInst.setMatrixAt(t, combMat);

    /* 8 high output floodlight bulbs */
    for (let row = -1; row <= 1; row += 2) {
      for (let col = -3; col <= 3; col += 2) {
        subDummy.position.set(col * 0.72, row * 0.45, 0.42);
        subDummy.rotation.set(0.4, 0, 0);
        subDummy.updateMatrix();
        combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);
        bulbInst.setMatrixAt(bulbCount++, combMat);
      }
    }

    /* Spotlights on primary towers */
    if (spotLightCount < 4) {
      const spot = new THREE.SpotLight(0xfffaed, 2.2);
      spot.position.set(d.poleX, d.poleHeight, d.poleZ);
      spot.target.position.set(d.trackPtX, 0, d.trackPtZ);
      spot.angle = 0.40;
      spot.penumbra = 0.65;
      spot.distance = 85;
      spot.decay = 2.0;
      group.add(spot);
      group.add(spot.target);
      spotLightCount++;
    }
  }

  poleInst.instanceMatrix.needsUpdate = true;
  platformInst.instanceMatrix.needsUpdate = true;
  headFrameInst.instanceMatrix.needsUpdate = true;
  bulbInst.instanceMatrix.needsUpdate = true;

  group.add(poleInst);
  group.add(platformInst);
  group.add(headFrameInst);
  group.add(bulbInst);

  return group;
}

let lightPoolTextureCached = null;
function getLightPoolTexture() {
  if (lightPoolTextureCached) return lightPoolTextureCached;
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 128;
  const ctx = canvas.getContext('2d');
  const grad = ctx.createRadialGradient(64, 64, 2, 64, 64, 62);
  grad.addColorStop(0, 'rgba(255, 246, 220, 0.45)');
  grad.addColorStop(0.35, 'rgba(255, 230, 175, 0.24)');
  grad.addColorStop(0.7, 'rgba(255, 210, 130, 0.08)');
  grad.addColorStop(1, 'rgba(255, 195, 90, 0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 128, 128);
  lightPoolTextureCached = new THREE.CanvasTexture(canvas);
  return lightPoolTextureCached;
}

let beamTextureCached = null;
function getBeamTexture() {
  if (beamTextureCached) return beamTextureCached;
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');
  /* Vertical beam gradient, bright at top near lamp, softly fading to 0 at bottom */
  const grad = ctx.createLinearGradient(0, 0, 0, 256);
  grad.addColorStop(0, 'rgba(255, 248, 220, 0.70)');
  grad.addColorStop(0.15, 'rgba(255, 238, 190, 0.40)');
  grad.addColorStop(0.45, 'rgba(255, 225, 160, 0.16)');
  grad.addColorStop(0.80, 'rgba(255, 215, 140, 0.04)');
  grad.addColorStop(1.0, 'rgba(255, 210, 120, 0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 64, 256);
  beamTextureCached = new THREE.CanvasTexture(canvas);
  return beamTextureCached;
}

let redHazeTextureCached = null;
function getRedHazeTexture() {
  if (redHazeTextureCached) return redHazeTextureCached;
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 128;
  const ctx = canvas.getContext('2d');
  /* Pure laser red pavement haze gradient matching car rear safety LED */
  const grad = ctx.createRadialGradient(64, 64, 0, 64, 64, 63);
  grad.addColorStop(0, 'rgba(255, 255, 255, 1.0)');
  grad.addColorStop(0.06, 'rgba(255, 210, 220, 0.98)');
  grad.addColorStop(0.14, 'rgba(255, 0, 34, 0.95)');
  grad.addColorStop(0.32, 'rgba(255, 0, 20, 0.70)');
  grad.addColorStop(0.58, 'rgba(220, 0, 25, 0.30)');
  grad.addColorStop(0.82, 'rgba(160, 0, 15, 0.08)');
  grad.addColorStop(1.0, 'rgba(80, 0, 10, 0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 128, 128);
  redHazeTextureCached = new THREE.CanvasTexture(canvas);
  return redHazeTextureCached;
}

let studFlareTextureCached = null;
function getStudFlareTexture() {
  if (studFlareTextureCached) return studFlareTextureCached;
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');
  /* High intensity optical starburst flare for direct line of sight road stud radiance */
  const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 31);
  grad.addColorStop(0, 'rgba(255, 255, 255, 1.0)');
  grad.addColorStop(0.10, 'rgba(255, 0, 34, 1.0)');
  grad.addColorStop(0.38, 'rgba(255, 0, 20, 0.70)');
  grad.addColorStop(0.72, 'rgba(200, 0, 15, 0.18)');
  grad.addColorStop(1.0, 'rgba(100, 0, 10, 0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 64, 64);
  studFlareTextureCached = new THREE.CanvasTexture(canvas);
  return studFlareTextureCached;
}

/*
 * I construct aesthetic trackside lighting poles along the sides of the asphalt.
 * Each pole features an angled aerodynamic luminaire head projecting across the racing line,
 * an angled atmospheric volumetric light cone, and a broad illumination puddle across the track.
 * My approach batches all mast columns, arms, heads, bulbs, light cones, and puddles into instanced meshes.
 */
function buildTracksideLightingPoles(track, trackCenter) {
  const group = new THREE.Group();
  const n = track.centerline.length;
  const step = Math.max(3, Math.floor(n / 32));

  const poleMat = new THREE.MeshStandardMaterial({ color: 0x0c1322, metalness: 0.85, roughness: 0.3 });
  const headMat = new THREE.MeshStandardMaterial({ color: 0x060a14, roughness: 0.4 });
  const bulbMat = new THREE.MeshBasicMaterial({ color: 0xfffaea });
  const accentMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff });

  const beamMat = new THREE.MeshBasicMaterial({
    map: getBeamTexture(),
    transparent: true,
    opacity: 0.35,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    fog: false,
  });

  const puddleMat = new THREE.MeshBasicMaterial({
    map: getLightPoolTexture(),
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.DoubleSide,
  });

  const mastGeo = new THREE.CylinderGeometry(0.14, 0.22, 8.5, 8);
  const armGeo = new THREE.BoxGeometry(2.4, 0.12, 0.18);
  const headGeo = new THREE.BoxGeometry(1.6, 0.22, 0.75);
  const bulbGeo = new THREE.BoxGeometry(1.4, 0.06, 0.55);
  const ringGeo = new THREE.CylinderGeometry(0.3, 0.35, 0.15, 8);
  const beamGeo = new THREE.CylinderGeometry(0.28, 4.6, 10.8, 16, 1, true);
  const puddleGeo = new THREE.PlaneGeometry(10.5, 8.5);

  const poleData = [];
  for (let i = 0; i < n; i += step) {
    const bL = track.barrier_left[i];
    const bR = track.barrier_right[i];

    const cp = track.centerline[i];
    const distL = Math.hypot(bL[0] - trackCenter.x, bL[1] - trackCenter.y);
    const distR = Math.hypot(bR[0] - trackCenter.x, bR[1] - trackCenter.y);
    const outerB = distL > distR ? bL : bR;

    const normX = outerB[0] - cp[0];
    const normY = outerB[1] - cp[1];
    const normLen = Math.hypot(normX, normY) || 1;

    const poleX = outerB[0] + (normX / normLen) * 2.2;
    const poleZ = -(outerB[1] + (normY / normLen) * 2.2);

    let minTrackDist = Infinity;
    for (let k = 0; k < n; k += 4) {
      const d = Math.hypot(poleX - track.centerline[k][0], -poleZ - track.centerline[k][1]);
      if (d < minTrackDist) minTrackDist = d;
    }
    if (minTrackDist < 8.0) continue;

    const trackPtX = track.centerline[i][0];
    const trackPtZ = -track.centerline[i][1];
    const lookAngle = Math.atan2(trackPtX - poleX, trackPtZ - poleZ);

    poleData.push({ poleX, poleZ, lookAngle, trackPtX, trackPtZ, i });
  }

  const numPoles = poleData.length;
  if (numPoles === 0) return group;

  const mastInst = new THREE.InstancedMesh(mastGeo, poleMat, numPoles);
  const ringInst = new THREE.InstancedMesh(ringGeo, accentMat, numPoles);
  const armInst = new THREE.InstancedMesh(armGeo, poleMat, numPoles);
  const headInst = new THREE.InstancedMesh(headGeo, headMat, numPoles);
  const bulbInst = new THREE.InstancedMesh(bulbGeo, bulbMat, numPoles);
  const beamInst = new THREE.InstancedMesh(beamGeo, beamMat, numPoles);
  const puddleInst = new THREE.InstancedMesh(puddleGeo, puddleMat, numPoles);

  const parentDummy = new THREE.Object3D();
  const subDummy = new THREE.Object3D();
  const combMat = new THREE.Matrix4();
  let lightCount = 0;

  for (let p = 0; p < numPoles; p++) {
    const d = poleData[p];
    parentDummy.position.set(d.poleX, 0, d.poleZ);
    parentDummy.rotation.set(0, d.lookAngle, 0);
    parentDummy.updateMatrix();

    /* 1. Mast */
    subDummy.position.set(0, 4.25, 0);
    subDummy.rotation.set(0, 0, 0);
    subDummy.updateMatrix();
    combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);
    mastInst.setMatrixAt(p, combMat);

    /* 2. Base ring */
    subDummy.position.set(0, 0.1, 0);
    subDummy.updateMatrix();
    combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);
    ringInst.setMatrixAt(p, combMat);

    /* 3. Cantilever arm */
    subDummy.position.set(0, 8.4, 1.2);
    subDummy.rotation.set(0.15, 0, 0);
    subDummy.updateMatrix();
    combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);
    armInst.setMatrixAt(p, combMat);

    /* 4. Luminaire head */
    subDummy.position.set(0, 8.45, 2.2);
    subDummy.rotation.set(0.66, 0, 0);
    subDummy.updateMatrix();
    combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);
    headInst.setMatrixAt(p, combMat);

    /* 5. Glowing LED bulb */
    subDummy.position.set(0, 8.35, 2.2);
    subDummy.rotation.set(0.66, 0, 0);
    subDummy.updateMatrix();
    combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);
    bulbInst.setMatrixAt(p, combMat);

    /* 6. Volumetric angled light beam */
    subDummy.position.set(0, 4.18, 5.50);
    subDummy.rotation.set(-0.66, 0, 0);
    subDummy.updateMatrix();
    combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);
    beamInst.setMatrixAt(p, combMat);

    /* 7. Radial light puddle */
    subDummy.position.set(0, 0.025, 8.8);
    subDummy.rotation.set(-Math.PI / 2, 0, 0);
    subDummy.updateMatrix();
    combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);
    puddleInst.setMatrixAt(p, combMat);

    /* 8. Track illumination spotlights */
    if (d.i % (step * 4) === 0 && lightCount < 4) {
      const headWorldX = d.poleX + Math.sin(d.lookAngle) * 2.2;
      const headWorldZ = d.poleZ + Math.cos(d.lookAngle) * 2.2;
      const targetWorldX = headWorldX + Math.sin(d.lookAngle) * 6.6;
      const targetWorldZ = headWorldZ + Math.cos(d.lookAngle) * 6.6;

      const downLight = new THREE.SpotLight(0xfffaea, 1.6);
      downLight.position.set(headWorldX, 8.35, headWorldZ);
      downLight.target.position.set(targetWorldX, 0, targetWorldZ);
      downLight.angle = 0.55;
      downLight.penumbra = 0.85;
      downLight.distance = 30;
      downLight.decay = 2.0;
      group.add(downLight);
      group.add(downLight.target);
      lightCount++;
    }
  }

  mastInst.instanceMatrix.needsUpdate = true;
  ringInst.instanceMatrix.needsUpdate = true;
  armInst.instanceMatrix.needsUpdate = true;
  headInst.instanceMatrix.needsUpdate = true;
  bulbInst.instanceMatrix.needsUpdate = true;
  beamInst.instanceMatrix.needsUpdate = true;
  puddleInst.instanceMatrix.needsUpdate = true;

  group.add(mastInst);
  group.add(ringInst);
  group.add(armInst);
  group.add(headInst);
  group.add(bulbInst);
  group.add(beamInst);
  group.add(puddleInst);

  return group;
}

/*
 * I construct glowing red hazy road studs embedded into the asphalt surface along track limits.
 * Each light features a physical metallic casing, intense red LED diode core, soft ground haze aura,
 * and vertical optical starburst flare quads ensuring direct line of sight radiance matching the car rear light.
 */
function buildAsphaltHazeLights(track) {
  const group = new THREE.Group();
  const n = track.centerline.length;
  const step = Math.max(1, Math.floor(n / 65));

  const groundHazeMat = new THREE.MeshBasicMaterial({
    map: getRedHazeTexture(),
    transparent: true,
    opacity: 0.96,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.DoubleSide,
    fog: false,
  });

  const flareMat = new THREE.MeshBasicMaterial({
    map: getStudFlareTexture(),
    transparent: true,
    opacity: 0.95,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.DoubleSide,
    fog: false,
  });

  const studMat = new THREE.MeshStandardMaterial({
    color: 0x151822,
    metalness: 0.85,
    roughness: 0.35,
  });

  const coreMat = new THREE.MeshBasicMaterial({
    color: 0xff0022,
  });

  const hazePositions = [];
  const hazeUvs = [];
  const hazeIndices = [];
  let hIdx = 0;

  const flarePositions = [];
  const flareUvs = [];
  const flareIndices = [];
  let fIdx = 0;

  const studTransforms = [];

  function addLightPoint(x, z) {
    /* 1. Ground haze quad lying flat on the asphalt surface with rich circular dispersion */
    const hSize = 2.4;
    const hHalf = hSize / 2;
    const yGround = 0.024;

    hazePositions.push(
      x - hHalf, yGround, z - hHalf,
      x + hHalf, yGround, z - hHalf,
      x + hHalf, yGround, z + hHalf,
      x - hHalf, yGround, z + hHalf
    );
    hazeUvs.push(0, 0, 1, 0, 1, 1, 0, 1);
    hazeIndices.push(hIdx, hIdx + 1, hIdx + 2, hIdx, hIdx + 2, hIdx + 3);
    hIdx += 4;

    /* 2. Direct line of sight vertical optical starburst flare quads */
    const fHalf = 0.22;
    const yBase = 0.02;
    const yPeak = 0.36;

    /* Flare Quad 1 along X axis */
    flarePositions.push(
      x - fHalf, yBase, z,
      x + fHalf, yBase, z,
      x + fHalf, yPeak, z,
      x - fHalf, yPeak, z
    );
    flareUvs.push(0, 0, 1, 0, 1, 1, 0, 1);
    flareIndices.push(fIdx, fIdx + 1, fIdx + 2, fIdx, fIdx + 2, fIdx + 3);
    fIdx += 4;

    /* Flare Quad 2 along Z axis */
    flarePositions.push(
      x, yBase, z - fHalf,
      x, yBase, z + fHalf,
      x, yPeak, z + fHalf,
      x, yPeak, z - fHalf
    );
    flareUvs.push(0, 0, 1, 0, 1, 1, 0, 1);
    flareIndices.push(fIdx, fIdx + 1, fIdx + 2, fIdx, fIdx + 2, fIdx + 3);
    fIdx += 4;

    /* 3. Physical road stud transform */
    const mat = new THREE.Matrix4();
    mat.setPosition(x, 0.024, z);
    studTransforms.push(mat);
  }

  for (let i = 0; i < n; i += step) {
    const cPt = track.centerline[i];

    /* Left asphalt boundary */
    const bL = track.left_boundary[i];
    const dxL = cPt[0] - bL[0];
    const dyL = cPt[1] - bL[1];
    const lenL = Math.hypot(dxL, dyL) || 1;
    const posLx = bL[0] + (dxL / lenL) * 0.35;
    const posLz = -(bL[1] + (dyL / lenL) * 0.35);
    addLightPoint(posLx, posLz);

    /* Right asphalt boundary */
    const bR = track.right_boundary[i];
    const dxR = cPt[0] - bR[0];
    const dyR = cPt[1] - bR[1];
    const lenR = Math.hypot(dxR, dyR) || 1;
    const posRx = bR[0] + (dxR / lenR) * 0.35;
    const posRz = -(bR[1] + (dyR / lenR) * 0.35);
    addLightPoint(posRx, posRz);
  }

  /* Assemble merged ground haze mesh */
  const hazeGeo = new THREE.BufferGeometry();
  hazeGeo.setAttribute('position', new THREE.Float32BufferAttribute(hazePositions, 3));
  hazeGeo.setAttribute('uv', new THREE.Float32BufferAttribute(hazeUvs, 2));
  hazeGeo.setIndex(hazeIndices);
  const hazeMesh = new THREE.Mesh(hazeGeo, groundHazeMat);
  group.add(hazeMesh);

  /* Assemble merged vertical optical starburst flare mesh */
  const flareGeo = new THREE.BufferGeometry();
  flareGeo.setAttribute('position', new THREE.Float32BufferAttribute(flarePositions, 3));
  flareGeo.setAttribute('uv', new THREE.Float32BufferAttribute(flareUvs, 2));
  flareGeo.setIndex(flareIndices);
  const flareMesh = new THREE.Mesh(flareGeo, flareMat);
  group.add(flareMesh);

  /* Instanced physical road studs embedded into the tarmac */
  const studCount = studTransforms.length;
  if (studCount > 0) {
    const studBaseGeo = new THREE.CylinderGeometry(0.12, 0.16, 0.03, 10);
    const studMesh = new THREE.InstancedMesh(studBaseGeo, studMat, studCount);
    const coreGeo = new THREE.CylinderGeometry(0.075, 0.075, 0.045, 10);
    const coreMesh = new THREE.InstancedMesh(coreGeo, coreMat, studCount);

    for (let idx = 0; idx < studCount; idx++) {
      studMesh.setMatrixAt(idx, studTransforms[idx]);
      coreMesh.setMatrixAt(idx, studTransforms[idx]);
    }
    studMesh.instanceMatrix.needsUpdate = true;
    coreMesh.instanceMatrix.needsUpdate = true;

    group.add(studMesh);
    group.add(coreMesh);
  }

  return group;
}

/*
 * I build spectator grandstands surrounding the entire circuit perimeter.
 * My approach places open terraced concrete bleachers, overhead floodlit canopy roofs,
 * glowing neon accents, and dense instanced low poly humanoid spectators cheering.
 */
function buildGrandstands(track, trackCenter) {
  const group = new THREE.Group();
  const n = track.centerline.length;
  const step = Math.max(6, Math.floor(n / 16));

  const concreteMat = new THREE.MeshStandardMaterial({ color: 0x4a5d78, roughness: 0.45, metalness: 0.3 });
  const wallMat = new THREE.MeshStandardMaterial({ color: 0x1e2c3d, roughness: 0.5, metalness: 0.4 });
  const roofMat = new THREE.MeshStandardMaterial({ color: 0x182433, roughness: 0.4, metalness: 0.5 });
  const neonMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff });
  const seatMatRed = new THREE.MeshStandardMaterial({ color: 0xef233c, roughness: 0.5, emissive: 0x55000a, emissiveIntensity: 0.4 });
  const seatMatBlue = new THREE.MeshStandardMaterial({ color: 0x2563eb, roughness: 0.5, emissive: 0x082050, emissiveIntensity: 0.4 });
  const pillarMat = new THREE.MeshStandardMaterial({ color: 0x64748b, metalness: 0.85, roughness: 0.25 });
  const balustradeMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff, transparent: true, opacity: 0.5 });
  const fasciaMat = new THREE.MeshStandardMaterial({
    color: 0x07111e,
    emissive: 0x00f0ff,
    emissiveIntensity: 0.35,
    metalness: 0.8,
  });
  const ceilingMat = new THREE.MeshBasicMaterial({ color: 0xfffae0 });

  /* Vibrant team merchandise colors with subtle emissive glow */
  const fanColors = [
    0x00f0ff,
    0xff7700,
    0xe10600,
    0x00a19c,
    0x2563eb,
    0xfacc15,
  ];

  const shirtMaterials = fanColors.map(c => new THREE.MeshStandardMaterial({
    color: c,
    roughness: 0.5,
    emissive: c,
    emissiveIntensity: 0.35,
  }));
  const skinMat = new THREE.MeshStandardMaterial({ color: 0xf8d7b8, roughness: 0.6 });
  const capMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.5 });

  /* Low poly humanoid anatomical geometries */
  const torsoGeo = new THREE.CylinderGeometry(0.18, 0.12, 0.40, 6);
  const headGeo = new THREE.SphereGeometry(0.115, 6, 6);
  const capGeo = new THREE.BoxGeometry(0.16, 0.04, 0.14);

  /* Grandstand architectural geometries */
  const platformGeo = new THREE.BoxGeometry(18.2, 0.35, 1.6);
  const riserGeo = new THREE.BoxGeometry(18.2, 1.0, 0.20);
  const backWallGeo = new THREE.BoxGeometry(18.4, 4.4, 0.35);
  const sideWallGeo = new THREE.BoxGeometry(0.25, 4.2, 6.2);
  const neonSideRunnerGeo = new THREE.BoxGeometry(0.12, 0.12, 6.4);
  const roofGeo = new THREE.BoxGeometry(19.2, 0.40, 9.6);
  const ceilingLightGeo = new THREE.BoxGeometry(18.6, 0.12, 8.8);
  const fasciaBoardGeo = new THREE.BoxGeometry(19.2, 0.90, 0.25);
  const neonRoofEdgeGeo = new THREE.BoxGeometry(19.2, 0.08, 0.15);
  const balustradeGeo = new THREE.BoxGeometry(18.2, 0.85, 0.08);
  const handrailGeo = new THREE.BoxGeometry(18.2, 0.08, 0.12);
  const pillarGeo = new THREE.CylinderGeometry(0.22, 0.22, 7.4, 8);
  const seatBenchGeo = new THREE.BoxGeometry(16.4, 0.24, 0.65);

  const margin = Math.max(16, Math.floor(n / 24));
  const step = Math.max(8, Math.floor(n / 16));
  const validStands = [];

  for (let i = margin; i < n - margin; i += step) {
    const cp = track.centerline[i];
    const bL = track.barrier_left[i];
    const bR = track.barrier_right[i];

    const distL = Math.hypot(bL[0] - trackCenter.x, bL[1] - trackCenter.y);
    const distR = Math.hypot(bR[0] - trackCenter.x, bR[1] - trackCenter.y);
    const outerP = distL > distR ? bL : bR;

    const normX = outerP[0] - cp[0];
    const normY = outerP[1] - cp[1];
    const normLen = Math.hypot(normX, normY) || 1;

    const posX = outerP[0] + (normX / normLen) * 20.0;
    const posZ = -(outerP[1] + (normY / normLen) * 20.0);

    let minTrackDist = Infinity;
    for (let k = 0; k < n; k += 4) {
      const d = Math.hypot(posX - track.centerline[k][0], -posZ - track.centerline[k][1]);
      if (d < minTrackDist) minTrackDist = d;
    }
    if (minTrackDist < 18.0) continue;

    const trackPtX = cp[0];
    const trackPtZ = -cp[1];
    const standAngle = Math.atan2(posX - trackPtX, posZ - trackPtZ);
    const totalAngle = standAngle + Math.PI;

    validStands.push({ posX, posZ, totalAngle });
  }

  const numStands = validStands.length;
  const fansPerRow = 16;
  const numRows = 4;
  const totalSeatedFans = numStands * numRows * fansPerRow;
  const totalStandingFans = 32;
  const totalFans = totalSeatedFans + totalStandingFans;

  /* Instanced meshes for all spectators: 6 body draw calls, 1 head, 1 cap */
  const bodyInstances = shirtMaterials.map(m => new THREE.InstancedMesh(torsoGeo, m, Math.ceil(totalFans / fanColors.length) + 64));
  const headInstance = new THREE.InstancedMesh(headGeo, skinMat, totalFans);
  const capInstance = new THREE.InstancedMesh(capGeo, capMat, totalFans);

  const bodyCounts = new Array(fanColors.length).fill(0);
  let headCount = 0;
  let capCount = 0;

  /* Instanced meshes for all grandstand architectural structures */
  const platformInst = new THREE.InstancedMesh(platformGeo, concreteMat, numStands * 4);
  const riserInst = new THREE.InstancedMesh(riserGeo, concreteMat, numStands * 4);
  const backWallInst = new THREE.InstancedMesh(backWallGeo, wallMat, numStands);
  const sideWallInst = new THREE.InstancedMesh(sideWallGeo, wallMat, numStands * 2);
  const neonSideRunnerInst = new THREE.InstancedMesh(neonSideRunnerGeo, neonMat, numStands * 2);
  const roofInst = new THREE.InstancedMesh(roofGeo, roofMat, numStands);
  const ceilingLightInst = new THREE.InstancedMesh(ceilingLightGeo, ceilingMat, numStands);
  const fasciaBoardInst = new THREE.InstancedMesh(fasciaBoardGeo, fasciaMat, numStands);
  const neonRoofEdgeInst = new THREE.InstancedMesh(neonRoofEdgeGeo, neonMat, numStands);
  const balustradeInst = new THREE.InstancedMesh(balustradeGeo, balustradeMat, numStands);
  const handrailInst = new THREE.InstancedMesh(handrailGeo, neonMat, numStands);
  const pillarInst = new THREE.InstancedMesh(pillarGeo, pillarMat, numStands * 4);
  const seatRedInst = new THREE.InstancedMesh(seatBenchGeo, seatMatRed, numStands * 2);
  const seatBlueInst = new THREE.InstancedMesh(seatBenchGeo, seatMatBlue, numStands * 2);

  let platformCount = 0, riserCount = 0, backWallCount = 0, sideWallCount = 0;
  let runnerCount = 0, roofCount = 0, ceilingCount = 0, fasciaCount = 0;
  let roofEdgeCount = 0, balustradeCount = 0, handrailCount = 0, pillarCount = 0;
  let seatRedCount = 0, seatBlueCount = 0;

  const parentDummy = new THREE.Object3D();
  const subDummy = new THREE.Object3D();
  const combMat = new THREE.Matrix4();
  const dummy = new THREE.Object3D();

  const tierSteps = [
    { y: 0.85, z: 2.4, riserH: 0.70, riserY: 0.35, riserZ: 3.1 },
    { y: 1.80, z: 0.9, riserH: 0.95, riserY: 1.32, riserZ: 1.6 },
    { y: 2.75, z: -0.6, riserH: 0.95, riserY: 2.27, riserZ: 0.1 },
    { y: 3.70, z: -2.1, riserH: 0.95, riserY: 3.22, riserZ: -1.4 }
  ];

  for (let stIdx = 0; stIdx < numStands; stIdx++) {
    const { posX, posZ, totalAngle } = validStands[stIdx];
    parentDummy.position.set(posX, 0, posZ);
    parentDummy.rotation.set(0, totalAngle, 0);
    parentDummy.updateMatrix();

    /* 1. Terraced concrete platforms and risers */
    for (let r = 0; r < tierSteps.length; r++) {
      const stepCfg = tierSteps[r];
      subDummy.position.set(0, stepCfg.y, stepCfg.z);
      subDummy.rotation.set(0, 0, 0);
      subDummy.scale.set(1, 1, 1);
      subDummy.updateMatrix();
      combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);
      platformInst.setMatrixAt(platformCount++, combMat);

      subDummy.position.set(0, stepCfg.riserY, stepCfg.riserZ);
      subDummy.scale.set(1, stepCfg.riserH, 1);
      subDummy.updateMatrix();
      combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);
      riserInst.setMatrixAt(riserCount++, combMat);
    }
    subDummy.scale.set(1, 1, 1);

    /* 2. Rear wall */
    subDummy.position.set(0, 3.8, -3.0);
    subDummy.rotation.set(0, 0, 0);
    subDummy.updateMatrix();
    combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);
    backWallInst.setMatrixAt(backWallCount++, combMat);

    /* 3. Flank side enclosure walls and neon side runners */
    for (const side of [-1, 1]) {
      subDummy.position.set(side * 9.1, 2.6, 0.3);
      subDummy.rotation.set(0, 0, 0);
      subDummy.updateMatrix();
      combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);
      sideWallInst.setMatrixAt(sideWallCount++, combMat);

      subDummy.position.set(side * 9.12, 4.6, 0.3);
      subDummy.rotation.set(-0.45, 0, 0);
      subDummy.updateMatrix();
      combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);
      neonSideRunnerInst.setMatrixAt(runnerCount++, combMat);
    }

    /* 4. Cantilever canopy roof and ceiling light */
    subDummy.position.set(0, 7.4, 0.2);
    subDummy.rotation.set(0.05, 0, 0);
    subDummy.updateMatrix();
    combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);
    roofInst.setMatrixAt(roofCount++, combMat);

    subDummy.position.set(0, 7.15, 0.2);
    subDummy.rotation.set(0.05, 0, 0);
    subDummy.updateMatrix();
    combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);
    ceilingLightInst.setMatrixAt(ceilingCount++, combMat);

    /* 5. Fascia board and neon roof edge */
    subDummy.position.set(0, 7.6, 4.9);
    subDummy.rotation.set(0, 0, 0);
    subDummy.updateMatrix();
    combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);
    fasciaBoardInst.setMatrixAt(fasciaCount++, combMat);

    subDummy.position.set(0, 7.18, 4.95);
    subDummy.updateMatrix();
    combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);
    neonRoofEdgeInst.setMatrixAt(roofEdgeCount++, combMat);

    /* 6. Balustrade and handrail */
    subDummy.position.set(0, 1.35, 3.25);
    subDummy.updateMatrix();
    combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);
    balustradeInst.setMatrixAt(balustradeCount++, combMat);

    subDummy.position.set(0, 1.78, 3.25);
    subDummy.updateMatrix();
    combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);
    handrailInst.setMatrixAt(handrailCount++, combMat);

    /* 7. Support pillars */
    for (const pX of [-7.5, -2.5, 2.5, 7.5]) {
      subDummy.position.set(pX, 3.7, -3.0);
      subDummy.rotation.set(0, 0, 0);
      subDummy.updateMatrix();
      combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);
      pillarInst.setMatrixAt(pillarCount++, combMat);
    }

    /* 8. Stadium seating benches and spectators */
    const cosA = Math.cos(totalAngle);
    const sinA = Math.sin(totalAngle);

    for (let row = 0; row < tierSteps.length; row++) {
      const stepCfg = tierSteps[row];
      const rowY = stepCfg.y + 0.18;
      const rowZ = stepCfg.z;

      subDummy.position.set(0, rowY + 0.12, rowZ - 0.15);
      subDummy.rotation.set(0, 0, 0);
      subDummy.updateMatrix();
      combMat.multiplyMatrices(parentDummy.matrix, subDummy.matrix);

      if (row % 2 === 0) {
        seatRedInst.setMatrixAt(seatRedCount++, combMat);
      } else {
        seatBlueInst.setMatrixAt(seatBlueCount++, combMat);
      }

      /* Spectator instances */
      for (let s = 0; s < fansPerRow; s++) {
        const fanLocalX = -7.0 + (s / (fansPerRow - 1)) * 14.0;
        const fanLocalY = rowY + 0.42;
        const fanLocalZ = rowZ - 0.05;

        const worldX = posX + (fanLocalX * cosA + fanLocalZ * sinA);
        const worldY = fanLocalY;
        const worldZ = posZ + (-fanLocalX * sinA + fanLocalZ * cosA);

        const colorIdx = (stIdx + row * 2 + s) % fanColors.length;

        /* Torso */
        dummy.position.set(worldX, worldY, worldZ);
        dummy.rotation.set(0, totalAngle, 0);
        dummy.updateMatrix();
        bodyInstances[colorIdx].setMatrixAt(bodyCounts[colorIdx]++, dummy.matrix);

        /* Head */
        dummy.position.set(worldX, worldY + 0.28, worldZ);
        dummy.updateMatrix();
        headInstance.setMatrixAt(headCount++, dummy.matrix);

        /* Cap */
        dummy.position.set(worldX, worldY + 0.35, worldZ);
        dummy.updateMatrix();
        capInstance.setMatrixAt(capCount++, dummy.matrix);
      }
    }
  }

  /* 9. Trackside standing audience along the barrier walls */
  for (let s = 0; s < totalStandingFans; s++) {
    const ptIdx = (s * 4) % n;
    const cp = track.centerline[ptIdx];
    const b = track.barrier_right[ptIdx];
    const normX = b[0] - cp[0];
    const normY = b[1] - cp[1];
    const normLen = Math.hypot(normX, normY) || 1;

    const fanX = b[0] + (normX / normLen) * 2.5;
    const fanZ = -(b[1] + (normY / normLen) * 2.5);
    const facingAngle = Math.atan2(cp[0] - fanX, -cp[1] - fanZ);

    const colorIdx = s % fanColors.length;

    dummy.position.set(fanX, 0.92, fanZ);
    dummy.rotation.set(0, facingAngle, 0);
    dummy.updateMatrix();
    bodyInstances[colorIdx].setMatrixAt(bodyCounts[colorIdx]++, dummy.matrix);

    dummy.position.set(fanX, 1.20, fanZ);
    dummy.updateMatrix();
    headInstance.setMatrixAt(headCount++, dummy.matrix);

    dummy.position.set(fanX, 1.26, fanZ);
    dummy.updateMatrix();
    capInstance.setMatrixAt(capCount++, dummy.matrix);
  }

  /* Update all structure instance buffers */
  platformInst.instanceMatrix.needsUpdate = true; group.add(platformInst);
  riserInst.instanceMatrix.needsUpdate = true; group.add(riserInst);
  backWallInst.instanceMatrix.needsUpdate = true; group.add(backWallInst);
  sideWallInst.instanceMatrix.needsUpdate = true; group.add(sideWallInst);
  neonSideRunnerInst.instanceMatrix.needsUpdate = true; group.add(neonSideRunnerInst);
  roofInst.instanceMatrix.needsUpdate = true; group.add(roofInst);
  ceilingLightInst.instanceMatrix.needsUpdate = true; group.add(ceilingLightInst);
  fasciaBoardInst.instanceMatrix.needsUpdate = true; group.add(fasciaBoardInst);
  neonRoofEdgeInst.instanceMatrix.needsUpdate = true; group.add(neonRoofEdgeInst);
  balustradeInst.instanceMatrix.needsUpdate = true; group.add(balustradeInst);
  handrailInst.instanceMatrix.needsUpdate = true; group.add(handrailInst);
  pillarInst.instanceMatrix.needsUpdate = true; group.add(pillarInst);
  seatRedInst.instanceMatrix.needsUpdate = true; group.add(seatRedInst);
  seatBlueInst.instanceMatrix.needsUpdate = true; group.add(seatBlueInst);

  /* Update all crowd instance buffers */
  for (let c = 0; c < fanColors.length; c++) {
    bodyInstances[c].count = bodyCounts[c];
    bodyInstances[c].instanceMatrix.needsUpdate = true;
    group.add(bodyInstances[c]);
  }
  headInstance.count = headCount;
  headInstance.instanceMatrix.needsUpdate = true;
  group.add(headInstance);

  capInstance.count = capCount;
  capInstance.instanceMatrix.needsUpdate = true;
  group.add(capInstance);

  return group;
}

/*
 * Master assembly function for the 3D track environment.
 * Here I orchestrate the asphalt surface, dark rubberized racing line groove laid down by slick tires,
 * gravel traps, royal blue runoff areas, safety barriers, curbs, gantry, lighting, and grandstands.
 */
function buildTrackScene(track) {
  const group = new THREE.Group();

  let cx = 0, cy = 0;
  for (const p of track.centerline) { cx += p[0]; cy += p[1]; }
  cx /= track.centerline.length;
  cy /= track.centerline.length;
  const trackCenter = { x: cx, y: cy };

  /* 1. Asphalt Road Surface with procedural grain bump map */
  const asphaltBump = getAsphaltTexture();
  const roadMat = new THREE.MeshStandardMaterial({
    color: 0x202632,
    roughness: 0.70,
    metalness: 0.22,
    bumpMap: asphaltBump,
    bumpScale: 0.02,
    side: THREE.DoubleSide,
  });
  group.add(buildRibbonMesh(track.left_boundary, track.right_boundary, roadMat, 0.015));

  /* 2. Dark Tire Rubber Racing Line Groove laid down by Pirelli slicks on the ideal racing line */
  const rubberPolyL = [];
  const rubberPolyR = [];
  const n = track.centerline.length;
  const RUBBER_WIDTH = 1.6;
  for (let i = 0; i < n; i++) {
    const next = (i + 1) % n;
    const dx = track.centerline[next][0] - track.centerline[i][0];
    const dy = track.centerline[next][1] - track.centerline[i][1];
    const len = Math.hypot(dx, dy) || 1;
    const nx = -dy / len;
    const ny = dx / len;
    const p = track.centerline[i];
    rubberPolyL.push([p[0] + nx * RUBBER_WIDTH, p[1] + ny * RUBBER_WIDTH]);
    rubberPolyR.push([p[0] - nx * RUBBER_WIDTH, p[1] - ny * RUBBER_WIDTH]);
  }
  const rubberMat = new THREE.MeshStandardMaterial({
    color: 0x0c1017,
    roughness: 0.88,
    metalness: 0.28,
    side: THREE.DoubleSide,
  });
  group.add(buildRibbonMesh(rubberPolyL, rubberPolyR, rubberMat, 0.018));

  /* 3. Runoff Areas: Volcanic Basalt Traps and Midnight Blue Runoffs */
  const gravelMat = new THREE.MeshStandardMaterial({
    color: 0x221e1a,
    roughness: 0.94,
    metalness: 0.06,
    side: THREE.DoubleSide,
  });
  const blueRunoffMat = new THREE.MeshStandardMaterial({
    color: 0x0a2c66,
    roughness: 0.78,
    metalness: 0.18,
    side: THREE.DoubleSide,
  });

  group.add(buildRibbonMesh(track.barrier_left, track.left_boundary, blueRunoffMat, 0.008));
  group.add(buildRibbonMesh(track.right_boundary, track.barrier_right, gravelMat, 0.008));

  /* 4. Track Boundary Edge Lines */
  group.add(buildEdgeLine(track.left_boundary, 0xffffff, 0.95));
  group.add(buildEdgeLine(track.right_boundary, 0xffffff, 0.95));

  /* 5. Dashed Centerline */
  const centerPoints = track.centerline.map(p => new THREE.Vector3(p[0], 0.025, -p[1]));
  centerPoints.push(centerPoints[0].clone());
  const centerGeo = new THREE.BufferGeometry().setFromPoints(centerPoints);
  const centerLine = new THREE.Line(
    centerGeo,
    new THREE.LineDashedMaterial({ color: 0xffffff, opacity: 0.35, transparent: true, dashSize: 3.0, gapSize: 3.5 })
  );
  centerLine.computeLineDistances();
  group.add(centerLine);

  /* 6. Safety Barrier Walls */
  const barrierHeight = 1.35;
  group.add(buildBarrierWall(track.barrier_left, barrierHeight, 0x161c28));
  group.add(buildBarrierWall(track.barrier_right, barrierHeight, 0x161c28));

  /* 7. Beveled 3D Apex Rumble Curbs */
  group.add(buildCurbs(track));

  /* 8. Distance Braking Boards */
  group.add(buildBrakeMarkers(track));

  /* 9. Start and Finish Gantry */
  group.add(buildStartFinishStructure(track, trackCenter));

  /* 10. Trackside Posters and Synapse Hoardings */
  group.add(buildTracksidePosters(track, trackCenter));

  /* 11. Aesthetic Trackside Lighting Poles along sides of asphalt */
  group.add(buildTracksideLightingPoles(track, trackCenter));

  /* 12. Stadium Perimeter Lights */
  group.add(buildStadiumLights(track, trackCenter));

  /* 13. Spectator Grandstands */
  group.add(buildGrandstands(track, trackCenter));

  /* 14. Hazy Red Asphalt Embedded Ground Lights */
  group.add(buildAsphaltHazeLights(track));

  return group;
}

/*
 * Phase 3: High Fidelity F1 Vehicle Models and Dynamic Carbon Brake Discs
 *
 * Here I assemble fully detailed Formula 1 car models using Three.js physical materials.
 * My approach models realistic aerodynamic surfaces, clearcoat paint shaders, glowing brake discs,
 * and a dynamic X ray overlap mode that turns liveries translucent when cars get too close,
 * preventing unsightly polygonal clipping during close wheel to wheel racing.
 */

/*
 * I generate custom car livery decals dynamically using an offscreen canvas.
 * This paints racing stripes, aerodynamic chevrons, driver numbers, and team sponsor markings.
 */
function createCarLiveryTexture(colorHex, carNumber, teamName) {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = colorHex;
  ctx.fillRect(0, 0, 512, 512);

  /* Carbon edge borders */
  ctx.fillStyle = '#0a0f1d';
  ctx.fillRect(0, 0, 56, 512);
  ctx.fillRect(456, 0, 56, 512);

  /* Center white and gold racing stripe */
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(246, 0, 20, 512);

  /* Aerodynamic speed chevrons */
  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.moveTo(56, 90); ctx.lineTo(116, 150); ctx.lineTo(56, 210);
  ctx.closePath();
  ctx.fill();

  ctx.beginPath();
  ctx.moveTo(456, 90); ctx.lineTo(396, 150); ctx.lineTo(456, 210);
  ctx.closePath();
  ctx.fill();

  /* Racing Number Roundel */
  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.arc(256, 310, 92, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = '#0f172a';
  ctx.lineWidth = 10;
  ctx.stroke();

  ctx.fillStyle = '#0f172a';
  ctx.font = '900 130px "Titillium Web", sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(String(carNumber), 256, 316);

  /* Sponsor Banner */
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(80, 75, 352, 55);
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 34px "Titillium Web", sans-serif';
  ctx.fillText(teamName, 256, 108);

  const tex = new THREE.CanvasTexture(canvas);
  tex.anisotropy = 8;
  return tex;
}

function createSoftShadowTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');
  const grad = ctx.createRadialGradient(128, 256, 40, 128, 256, 220);
  grad.addColorStop(0, 'rgba(0, 0, 0, 0.72)');
  grad.addColorStop(0.45, 'rgba(0, 0, 0, 0.38)');
  grad.addColorStop(0.85, 'rgba(0, 0, 0, 0.12)');
  grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 256, 512);
  return new THREE.CanvasTexture(canvas);
}

/*
 * I construct a detailed Formula 1 car model from individual aerodynamic components:
 * 1. Aerodynamic nose cone with sponsor decals
 * 2. Cockpit monocoque with driver helmet and steering wheel mini display
 * 3. Sculpted sidepods with dark intake vents
 * 4. Engine airbox and stabilizing shark fin
 * 5. Titanium halo cockpit safety hoop and side mirrors
 * 6. Multi element front wing with endplate canards
 * 7. Swan neck rear wing assembly with DRS flap actuator pod
 * 8. Underbody rear diffuser and central exhaust outlet
 * 9. Pulsing red rain safety LED flasher
 * 10. Wheels, rims, painted calipers, and glowing carbon brake discs
 */
function buildF1CarModel(liveryColor, accentColor, carNumber = '1') {
  const carGroup = new THREE.Group();
  const L = 4.8;
  const W = 2.0;

  const chassisGroup = new THREE.Group();
  carGroup.add(chassisGroup);

  const bodyMaterials = [];
  const edgeMaterials = [];
  const wireMaterials = [];
  const brakeDiscs = [];

  function makeBodyMat(color, roughness = 0.18, metalness = 0.82) {
    const m = new THREE.MeshPhysicalMaterial({
      color,
      roughness,
      metalness,
      clearcoat: 0.95,
      clearcoatRoughness: 0.08,
      transparent: true,
      opacity: 1.0,
    });
    bodyMaterials.push(m);
    return m;
  }

  const liveryMat = makeBodyMat(liveryColor, 0.15, 0.85);
  const carbonMat = makeBodyMat(0x0f172a, 0.45, 0.6);
  const accentMat = makeBodyMat(accentColor, 0.2, 0.75);
  const haloMat = makeBodyMat(0x1e293b, 0.3, 0.9);
  const visorMat = makeBodyMat(0x05070a, 0.05, 0.95);

  const tireMat = new THREE.MeshStandardMaterial({ color: 0x171717, roughness: 0.85, metalness: 0.05 });
  const sidewallMat = new THREE.MeshBasicMaterial({ color: 0xfacc15 }); /* Pirelli Yellow Soft compound sidewall stripe */
  const rimMat = new THREE.MeshStandardMaterial({ color: liveryColor, roughness: 0.2, metalness: 0.9 });

  function addChassisComponent(geo, mat, pos, rot = null) {
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.copy(pos);
    if (rot) mesh.rotation.copy(rot);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    chassisGroup.add(mesh);

    /* Glowing edge outlines for overlap detection */
    const edgeGeo = new THREE.EdgesGeometry(geo);
    const edgeMat = new THREE.LineBasicMaterial({
      color: liveryColor,
      transparent: true,
      opacity: 0.0,
      linewidth: 2,
    });
    edgeMaterials.push(edgeMat);
    const edgeLine = new THREE.LineSegments(edgeGeo, edgeMat);
    edgeLine.position.copy(pos);
    if (rot) edgeLine.rotation.copy(rot);
    chassisGroup.add(edgeLine);

    /* Wireframe matrix activated during X Ray overlap mode to reveal the internal engineering */
    const wireMat = new THREE.MeshBasicMaterial({
      color: liveryColor,
      wireframe: true,
      transparent: true,
      opacity: 0.0,
    });
    wireMaterials.push(wireMat);
    const wireMesh = new THREE.Mesh(geo, wireMat);
    wireMesh.position.copy(pos);
    if (rot) wireMesh.rotation.copy(rot);
    chassisGroup.add(wireMesh);

    return mesh;
  }

  /* 1. Aerodynamic Nose Cone */
  const noseShape = new THREE.Shape();
  noseShape.moveTo(-0.28, 0);
  noseShape.lineTo(0.28, 0);
  noseShape.lineTo(0.12, L * 0.44);
  noseShape.lineTo(-0.12, L * 0.44);
  noseShape.closePath();
  const noseExtrude = { depth: 0.32, bevelEnabled: true, bevelThickness: 0.04, bevelSize: 0.04 };
  const noseGeo = new THREE.ExtrudeGeometry(noseShape, noseExtrude);
  noseGeo.rotateX(-Math.PI / 2);
  addChassisComponent(noseGeo, liveryMat, new THREE.Vector3(0, 0.16, 0));

  const noseTip = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.15, 0.3), carbonMat);
  noseTip.position.set(0, 0.22, L * 0.44 + 0.1);
  chassisGroup.add(noseTip);

  /* 1b. Procedural Nose Livery Decal for sponsors and racing number */
  const decalGeo = new THREE.PlaneGeometry(0.44, 1.25);
  const colorHexStr = liveryColor === 0x00f0ff ? '#00f0ff' : '#ff5500';
  const teamTitle = carNumber === '1' ? 'SYNAPSE AI' : 'SYNAPSE GP';
  const liveryTex = createCarLiveryTexture(colorHexStr, carNumber, teamTitle);
  const decalMat = new THREE.MeshStandardMaterial({
    map: liveryTex,
    transparent: true,
    roughness: 0.2,
    metalness: 0.7,
    polygonOffset: true,
    polygonOffsetFactor: -1,
  });
  const decalMesh = new THREE.Mesh(decalGeo, decalMat);
  decalMesh.rotation.x = -Math.PI / 2 + 0.15;
  decalMesh.position.set(0, 0.36, L * 0.21);
  chassisGroup.add(decalMesh);

  /* 2. Cockpit Tub and Monocoque */
  const tubGeo = new THREE.BoxGeometry(0.78, 0.48, L * 0.44);
  addChassisComponent(tubGeo, carbonMat, new THREE.Vector3(0, 0.36, -L * 0.02));

  /* 3. Sculpted Sidepods with dark intake vents */
  const sidepodGeo = new THREE.BoxGeometry(0.42, 0.42, L * 0.42);
  addChassisComponent(sidepodGeo, liveryMat, new THREE.Vector3(-0.55, 0.34, -L * 0.06));
  addChassisComponent(sidepodGeo, liveryMat, new THREE.Vector3(0.55, 0.34, -L * 0.06));

  const intakeGeo = new THREE.PlaneGeometry(0.36, 0.34);
  const intakeMat = new THREE.MeshBasicMaterial({ color: 0x090d16 });
  const leftIntake = new THREE.Mesh(intakeGeo, intakeMat);
  leftIntake.position.set(-0.55, 0.34, -L * 0.06 + L * 0.21 + 0.01);
  chassisGroup.add(leftIntake);

  const rightIntake = new THREE.Mesh(intakeGeo, intakeMat);
  rightIntake.position.set(0.55, 0.34, -L * 0.06 + L * 0.21 + 0.01);
  chassisGroup.add(rightIntake);

  /* 4. Engine Airbox and Shark Fin */
  const airbox = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.35, 0.6), carbonMat);
  airbox.position.set(0, 0.72, -0.2);
  chassisGroup.add(airbox);

  const finShape = new THREE.Shape();
  finShape.moveTo(0, 0);
  finShape.lineTo(0, L * 0.34);
  finShape.lineTo(0.38, 0);
  finShape.closePath();
  const finGeo = new THREE.ExtrudeGeometry(finShape, { depth: 0.04, bevelEnabled: false });
  finGeo.rotateY(Math.PI / 2);
  addChassisComponent(finGeo, accentMat, new THREE.Vector3(0.02, 0.58, -L * 0.4));

  /* 5. Cockpit Halo, Driver Helmet, and Mirrors */
  const haloGeo = new THREE.TorusGeometry(0.25, 0.04, 8, 16, Math.PI);
  const haloEuler = new THREE.Euler(Math.PI / 2, 0, 0);
  addChassisComponent(haloGeo, haloMat, new THREE.Vector3(0, 0.65, 0.15), haloEuler);

  const helmetGeo = new THREE.SphereGeometry(0.15, 16, 16);
  addChassisComponent(helmetGeo, accentMat, new THREE.Vector3(0, 0.56, 0.1));

  const visorGeo = new THREE.BoxGeometry(0.2, 0.07, 0.1);
  addChassisComponent(visorGeo, visorMat, new THREE.Vector3(0, 0.56, 0.2));

  /* Aerodynamic Wing Mirrors */
  const mirrorGlassMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, metalness: 0.95, roughness: 0.05 });
  for (const side of [-1, 1]) {
    const mirrorStalk = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.014, 0.18), carbonMat);
    mirrorStalk.position.set(side * 0.42, 0.57, 0.16);
    mirrorStalk.rotation.z = side * -0.45;
    chassisGroup.add(mirrorStalk);

    const mirrorHousing = new THREE.Mesh(new THREE.BoxGeometry(0.13, 0.07, 0.05), liveryMat);
    mirrorHousing.position.set(side * 0.47, 0.63, 0.16);
    chassisGroup.add(mirrorHousing);

    const mirrorGlass = new THREE.Mesh(new THREE.PlaneGeometry(0.10, 0.055), mirrorGlassMat);
    mirrorGlass.position.set(side * 0.47, 0.63, 0.134);
    chassisGroup.add(mirrorGlass);
  }

  /* 6. Cockpit Steering Wheel with Mini LCD Screen */
  const dashGeo = new THREE.BoxGeometry(0.28, 0.14, 0.04);
  const dashMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.6 });
  const dash = new THREE.Mesh(dashGeo, dashMat);
  dash.position.set(0, 0.44, 0.16);
  dash.rotation.x = -Math.PI / 5;
  chassisGroup.add(dash);

  const screenGeo = new THREE.PlaneGeometry(0.14, 0.07);
  const screenMat = new THREE.MeshBasicMaterial({ color: 0x00e676 });
  const screen = new THREE.Mesh(screenGeo, screenMat);
  screen.position.set(0, 0.45, 0.185);
  screen.rotation.x = -Math.PI / 5;
  chassisGroup.add(screen);

  /* 7. Multi Tier Front Wing Assembly with Endplate Canards */
  const fwMainGeo = new THREE.BoxGeometry(W * 0.94, 0.07, 0.42);
  addChassisComponent(fwMainGeo, carbonMat, new THREE.Vector3(0, 0.12, L * 0.46));

  const fwFlapGeo = new THREE.BoxGeometry(W * 0.9, 0.05, 0.22);
  addChassisComponent(fwFlapGeo, accentMat, new THREE.Vector3(0, 0.17, L * 0.44));

  const fwUpperFlap = new THREE.BoxGeometry(W * 0.86, 0.035, 0.14);
  addChassisComponent(fwUpperFlap, liveryMat, new THREE.Vector3(0, 0.22, L * 0.42));

  for (const side of [-1, 1]) {
    const epGeo = new THREE.BoxGeometry(0.04, 0.26, 0.48);
    addChassisComponent(epGeo, liveryMat, new THREE.Vector3(side * (W * 0.47), 0.2, L * 0.46));

    /* Outer footplate vortex canard */
    const canard = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.02, 0.22), carbonMat);
    canard.position.set(side * (W * 0.49), 0.14, L * 0.46);
    chassisGroup.add(canard);
  }

  /* 8. Rear Wing Assembly with Swan Neck Pylons and DRS Actuator */
  const rwMainGeo = new THREE.BoxGeometry(W * 0.74, 0.06, 0.34);
  addChassisComponent(rwMainGeo, carbonMat, new THREE.Vector3(0, 0.78, -L * 0.44));

  const rwFlapGeo = new THREE.BoxGeometry(W * 0.72, 0.05, 0.2);
  addChassisComponent(rwFlapGeo, liveryMat, new THREE.Vector3(0, 0.85, -L * 0.46));

  /* Dual Swan Neck Mounts */
  for (const side of [-1, 1]) {
    const pylon = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.36, 0.14), carbonMat);
    pylon.position.set(side * 0.14, 0.70, -L * 0.40);
    pylon.rotation.x = -0.25;
    chassisGroup.add(pylon);
  }

  /* DRS Actuator Pod */
  const drsPod = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.07, 0.15), carbonMat);
  drsPod.position.set(0, 0.87, -L * 0.45);
  chassisGroup.add(drsPod);

  for (const side of [-1, 1]) {
    const rwEpGeo = new THREE.BoxGeometry(0.04, 0.48, 0.44);
    addChassisComponent(rwEpGeo, carbonMat, new THREE.Vector3(side * (W * 0.37), 0.7, -L * 0.44));

    /* Rear wing endplate trailing edge red aero LED lights */
    const epLightMat = new THREE.MeshBasicMaterial({ color: 0xff1133 });
    const epEdgeLight = new THREE.Mesh(new THREE.BoxGeometry(0.015, 0.44, 0.02), epLightMat);
    epEdgeLight.position.set(side * (W * 0.37), 0.7, -L * 0.44 - 0.22);
    chassisGroup.add(epEdgeLight);
  }

  /* 8b. Underbody Rear Diffuser, Beam Wing, and Central Exhaust */
  const diffuserFloor = new THREE.Mesh(new THREE.BoxGeometry(W * 0.76, 0.05, 0.75), carbonMat);
  diffuserFloor.position.set(0, 0.12, -L * 0.38);
  chassisGroup.add(diffuserFloor);

  /* 4 sculpted carbon vertical vortex generator strakes */
  for (let s = -2; s <= 2; s++) {
    if (s === 0) continue;
    const strake = new THREE.Mesh(new THREE.BoxGeometry(0.025, 0.18, 0.7), carbonMat);
    strake.position.set(s * 0.19, 0.16, -L * 0.38);
    chassisGroup.add(strake);
  }

  /* Carbon rear crash attenuator structure extending backwards */
  const crashStructure = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.16, 0.35), carbonMat);
  crashStructure.position.set(0, 0.24, -L * 0.46);
  chassisGroup.add(crashStructure);

  /* Dual element aerodynamic beam wing mounted between diffuser and upper rear wing */
  const beamWingLower = new THREE.Mesh(new THREE.BoxGeometry(W * 0.68, 0.025, 0.16), carbonMat);
  beamWingLower.position.set(0, 0.46, -L * 0.43);
  chassisGroup.add(beamWingLower);

  const beamWingUpper = new THREE.Mesh(new THREE.BoxGeometry(W * 0.64, 0.02, 0.12), accentMat);
  beamWingUpper.position.set(0, 0.52, -L * 0.42);
  chassisGroup.add(beamWingUpper);

  /* Blued titanium exhaust pipe with internal molten heat core and flame glow light */
  const exhaustMat = new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.95, roughness: 0.15 });
  const exhaust = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.065, 0.22, 16), exhaustMat);
  exhaust.rotation.x = Math.PI / 2;
  exhaust.position.set(0, 0.36, -L * 0.46);
  chassisGroup.add(exhaust);

  const heatRingMat = new THREE.MeshStandardMaterial({ color: 0x3b82f6, metalness: 0.95, roughness: 0.1 });
  const heatRing = new THREE.Mesh(new THREE.CylinderGeometry(0.052, 0.052, 0.04, 16), heatRingMat);
  heatRing.rotation.x = Math.PI / 2;
  heatRing.position.set(0, 0.36, -L * 0.46 - 0.09);
  chassisGroup.add(heatRing);

  const exhaustCoreMat = new THREE.MeshBasicMaterial({ color: 0xff8800 });
  const exhaustCore = new THREE.Mesh(new THREE.CylinderGeometry(0.038, 0.038, 0.12, 12), exhaustCoreMat);
  exhaustCore.rotation.x = Math.PI / 2;
  exhaustCore.position.set(0, 0.36, -L * 0.46 - 0.06);
  chassisGroup.add(exhaustCore);

  const exhaustGlow = new THREE.PointLight(0xff6600, 1.2, 3.5, 2);
  exhaustGlow.position.set(0, 0.36, -L * 0.48);
  chassisGroup.add(exhaustGlow);

  /* 9. Pulsing Red Rain Safety LED Matrix with Dedicated Glow PointLight */
  const ledHousingMat = new THREE.MeshStandardMaterial({ color: 0x0a0a0a, roughness: 0.8 });
  const ledHousing = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.15, 0.04), ledHousingMat);
  ledHousing.position.set(0, 0.4, -L * 0.48);
  chassisGroup.add(ledHousing);

  const ledGeo = new THREE.BoxGeometry(0.14, 0.11, 0.03);
  const ledMat = new THREE.MeshBasicMaterial({ color: 0xff0022 });
  const rainLed = new THREE.Mesh(ledGeo, ledMat);
  rainLed.position.set(0, 0.4, -L * 0.48 - 0.02);
  chassisGroup.add(rainLed);

  const rainLight = new THREE.PointLight(0xff0022, 1.8, 4.0, 2);
  rainLight.position.set(0, 0.4, -L * 0.50);
  chassisGroup.add(rainLight);

  /* 10. Wheels, Rims, Painted Calipers, and Glowing Carbon Brake Discs */
  const wheelRadius = 0.34;
  const wheelWidth = 0.3;
  const wheelGeo = new THREE.CylinderGeometry(wheelRadius, wheelRadius, wheelWidth, 24);
  wheelGeo.rotateZ(Math.PI / 2);

  const sidewallGeo = new THREE.RingGeometry(wheelRadius * 0.65, wheelRadius * 0.82, 24);
  const rimGeo = new THREE.CylinderGeometry(wheelRadius * 0.55, wheelRadius * 0.55, wheelWidth * 1.02, 16);
  rimGeo.rotateZ(Math.PI / 2);

  const discGeo = new THREE.CylinderGeometry(wheelRadius * 0.52, wheelRadius * 0.52, wheelWidth * 0.35, 16);
  discGeo.rotateZ(Math.PI / 2);

  const wheelPositions = [
    { x: -W * 0.48, z: L * 0.32, isFront: true },
    { x: W * 0.48, z: L * 0.32, isFront: true },
    { x: -W * 0.48, z: -L * 0.32, isFront: false },
    { x: W * 0.48, z: -L * 0.32, isFront: false },
  ];

  carGroup.userData.wheels = [];
  carGroup.userData.frontWheels = [];
  carGroup.userData.brakeDiscs = brakeDiscs;
  carGroup.userData.chassis = chassisGroup;
  carGroup.userData.rainLed = rainLed;
  carGroup.userData.rainLight = rainLight;
  carGroup.userData.exhaustCore = exhaustCore;
  carGroup.userData.exhaustGlow = exhaustGlow;
  carGroup.userData.bodyMaterials = bodyMaterials;
  carGroup.userData.edgeMaterials = edgeMaterials;
  carGroup.userData.wireMaterials = wireMaterials;

  const wheelsParent = new THREE.Group();
  carGroup.add(wheelsParent);

  for (const wp of wheelPositions) {
    const wheelAnchor = new THREE.Group();
    wheelAnchor.position.set(wp.x, wheelRadius, wp.z);

    const wheelMesh = new THREE.Mesh(wheelGeo, tireMat);
    wheelMesh.castShadow = true;

    /* Pirelli yellow sidewall rings */
    const sideDir = Math.sign(wp.x);
    const sidewall = new THREE.Mesh(sidewallGeo, sidewallMat);
    sidewall.position.set(sideDir * (wheelWidth * 0.51), 0, 0);
    sidewall.rotation.y = sideDir * Math.PI / 2;
    wheelMesh.add(sidewall);

    /* Wheel rim */
    const rimMesh = new THREE.Mesh(rimGeo, rimMat);
    wheelMesh.add(rimMesh);

    /* Directional Center Lock Wheel Nut, red on left and blue on right */
    const nutColor = sideDir < 0 ? 0xef4444 : 0x3b82f6;
    const nutMat = new THREE.MeshStandardMaterial({ color: nutColor, metalness: 0.95, roughness: 0.15 });
    const nutMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.065, 0.065, wheelWidth * 0.18, 6), nutMat);
    nutMesh.position.set(sideDir * (wheelWidth * 0.54), 0, 0);
    nutMesh.rotation.z = Math.PI / 2;
    wheelMesh.add(nutMesh);

    /* Glowing carbon brake disc that flares into hot glowing orange under heavy braking deceleration */
    const discMat = new THREE.MeshStandardMaterial({
      color: 0x1f2937,
      emissive: 0x000000,
      emissiveIntensity: 0.0,
      roughness: 0.45,
      metalness: 0.85,
    });
    const discMesh = new THREE.Mesh(discGeo, discMat);
    wheelAnchor.add(discMesh);
    brakeDiscs.push(discMat);

    /* Painted Brake Caliper */
    const caliperMat = new THREE.MeshStandardMaterial({
      color: liveryColor === 0x00f0ff ? 0x00f0ff : 0xd90429,
      roughness: 0.3,
      metalness: 0.7,
    });
    const caliperMesh = new THREE.Mesh(new THREE.BoxGeometry(wheelWidth * 0.45, 0.14, 0.19), caliperMat);
    caliperMesh.position.set(sideDir * (wheelWidth * 0.16), 0.12, 0);
    wheelAnchor.add(caliperMesh);

    wheelAnchor.add(wheelMesh);
    wheelsParent.add(wheelAnchor);

    carGroup.userData.wheels.push(wheelMesh);
    if (wp.isFront) {
      carGroup.userData.frontWheels.push(wheelAnchor);
    }
  }

  /* 11. Soft Radial Ambient Contact Shadow */
  const shadowGeo = new THREE.PlaneGeometry(W * 1.35, L * 1.25);
  const shadowTex = createSoftShadowTexture();
  const shadowMat = new THREE.MeshBasicMaterial({
    map: shadowTex,
    transparent: true,
    opacity: 0.85,
    depthWrite: false,
  });
  const shadow = new THREE.Mesh(shadowGeo, shadowMat);
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.y = 0.025;
  carGroup.add(shadow);

  return carGroup;
}

/* Main Application State */

let scene, camera, renderer3d, clock;
let mainSunLight = null, hemiLight = null, ambientLight = null, groundMesh = null;
let replayA = null, replayB = null;
let carModelA = null, carModelB = null;
let trackGroup = null;
let spectatorFlags = [];

/* Live telecast state variables */
let liveMode = false;
let liveSocket = null;
let liveRunId = null;
let liveLastSeq = -1;
let liveCars = [];
let liveEnded = false;
let liveLastPacketAt = 0;
let liveStaleShown = false;
let livePhysicsHz = 30;

/* Settings and playback modes */
let playing = false;
let playbackSpeed = 1.0;
let currentTime = 0;
let maxTime = 0;
let cameraMode = 'battle'; /* Available camera modes: battle, carA, carB, cockpit, top, side */
let laneSeparation = true;

/* Incident and overtake tracking state */
let lastLeader = null;
let lastOffTrackA = false;
let lastOffTrackB = false;

/* Camera smooth dampening vectors */
const camTargetPos = new THREE.Vector3();
const camTargetLook = new THREE.Vector3();
const camCurrentPos = new THREE.Vector3();
const camCurrentLook = new THREE.Vector3();

/* Track bounding box coordinates */
const trackBounds = { minX: 0, maxX: 0, minZ: 0, maxZ: 0 };

/* DOM elements references */
const loadingOverlay = document.getElementById('loading-overlay');
const loadStatus = document.getElementById('load-status');
const filePicker = document.getElementById('file-picker');
const fileA = document.getElementById('file-a');
const fileB = document.getElementById('file-b');
const btnLoad = document.getElementById('btn-load');
const btnQuickLoad = document.getElementById('btn-quick-load');
const btnClosePicker = document.getElementById('btn-close-picker');
const btnOpenPicker = document.getElementById('btn-open-picker');
const pickerError = document.getElementById('picker-error');
const canvas = document.getElementById('race-canvas');

const btnPlay = document.getElementById('btn-play');
const btnRestart = document.getElementById('btn-restart');
const btnStepBack = document.getElementById('btn-step-back');
const btnStepFwd = document.getElementById('btn-step-fwd');
const btnLanes = document.getElementById('btn-lanes');
const btnAudio = document.getElementById('btn-audio');
const scrubber = document.getElementById('scrubber');
const timeCurrent = document.getElementById('time-current');
const timeTotal = document.getElementById('time-total');
const progressBar = document.getElementById('race-progress-bar');
const broadcastAlert = document.getElementById('broadcast-alert');
const alertText = document.getElementById('alert-text');
const overlapIndicator = document.getElementById('overlap-indicator');
const towerSector = document.getElementById('tower-sector');
const towerSession = document.getElementById('tower-session');
const liveUrl = document.getElementById('live-url');
const btnLiveConnect = document.getElementById('btn-live-connect');

/* Winner victory podium modal elements */
const winnerModal = document.getElementById('winner-modal');
const btnCloseWinner = document.getElementById('btn-close-winner');
const winnerName = document.getElementById('winner-name');
const winnerTimeVal = document.getElementById('winner-time-val');
const winnerP1Ind = document.getElementById('winner-p1-ind');
const podP1Name = document.getElementById('pod-p1-name');
const podP1Gap = document.getElementById('pod-p1-gap');
const winnerP2Ind = document.getElementById('winner-p2-ind');
const podP2Name = document.getElementById('pod-p2-name');
const podP2Gap = document.getElementById('pod-p2-gap');
const btnReplayWinner = document.getElementById('btn-replay-winner');
const btnSwitchTrackWinner = document.getElementById('btn-switch-track-winner');
const trackSwitcher = document.getElementById('track-switcher');

/* Hero section and cinematic director elements */
const heroSection = document.getElementById('hero-section');
const heroCamLabel = document.getElementById('hero-cam-label');
const cutFlashElem = document.getElementById('broadcast-cut-flash');
const btnLaunchGP = document.getElementById('btn-launch-gp');
const btnHeroShow = document.getElementById('btn-hero-show');
const heroSelectRacerA = document.getElementById('hero-select-racer-a');
const heroSelectRacerB = document.getElementById('hero-select-racer-b');
const heroFileA = document.getElementById('hero-file-a');
const heroFileB = document.getElementById('hero-file-b');
const heroFileNameA = document.getElementById('hero-file-name-a');
const heroFileNameB = document.getElementById('hero-file-name-b');


function liveInitialFrame(track) {
  const point = track.centerline?.[0] || [0, 0];
  return {
    x: Number(point[0]) || 0,
    y: Number(point[1]) || 0,
    heading: Number(track.heading?.[0]) || 0,
    speed: 0,
    lateral_velocity: 0,
    yaw_rate: 0,
    steer: 0,
    steering_command: 0,
    throttle_brake: 0,
    off_track: false,
    events: [],
    tick: 0,
    terminated: false,
    truncated: false,
  };
}

function liveReplayShell(track, entrant) {
  const participantId = entrant?.participant_id || 'Unknown Racer';
  return {
    participantId,
    tickHz: livePhysicsHz,
    car: { length_m: 4.8, width_m: 2.0 },
    track,
    frames: [liveInitialFrame(track)],
    result: {},
  };
}

function setLiveControls(enabled) {
  for (const control of [btnPlay, btnRestart, btnStepBack, btnStepFwd, scrubber, trackSwitcher]) {
    if (control) control.disabled = enabled;
  }
  if (towerSession) towerSession.textContent = enabled ? 'LIVE' : 'RACE';
}

function handleLiveHello(message) {
  liveMode = true;
  liveEnded = false;
  liveRunId = message.run_id;
  liveLastSeq = -1;
  livePhysicsHz = message.physics_hz;
  liveCars = [];
  winnerShownForCurrentRace = false;
  currentTime = 0;
  maxTime = 0;
  lastLeader = null;
  lastOffTrackA = false;
  lastOffTrackB = false;
  replayA = liveReplayShell(message.track, message.entrants[0]);
  replayB = liveReplayShell(message.track, message.entrants[1] || { participant_id: 'Waiting' });
  replayA.tickHz = livePhysicsHz;
  replayB.tickHz = livePhysicsHz;
  setLiveControls(true);
  if (heroSection) heroSection.classList.add('hidden');
  document.body.classList.remove('hero-active');
  isHeroMode = false;
  startRace();
  if (timeTotal) timeTotal.textContent = 'LIVE';
  showAlert('LIVE PYTHON RACE CONNECTED — GREEN FLAG', 'info', 3000);
}

function handleLiveState(message) {
  liveCars = message.cars.slice(0, 2).map((car) => ({
    ...car,
    finished: car.status === 'finished',
    tick: message.tick,
  }));
  currentTime = message.sim_time_s;
  maxTime = Math.max(maxTime, currentTime);
  liveLastPacketAt = performance.now();
  liveStaleShown = false;
  if (towerSession) towerSession.textContent = 'LIVE';
  for (const [index, car] of liveCars.entries()) {
    const target = index === 0 ? replayA : replayB;
    if (!target) continue;
    target.participantId = car.label || car.participant_id;
    target.frames = [{ ...car, tick: message.tick }];
    target.result = { status: car.status, steps: message.tick, elapsed_s: message.sim_time_s };
  }
}

function handleLiveEnd(message) {
  liveEnded = true;
  currentTime = message.sim_time_s;
  maxTime = Math.max(maxTime, currentTime);
  for (const [index, result] of message.results.entries()) {
    const target = index === 0 ? replayA : replayB;
    if (!target) continue;
    target.result = {
      status: result.status,
      steps: result.steps,
      elapsed_s: message.sim_time_s,
    };
  }
  showAlert('LIVE RACE FINISHED — OFFICIAL RESULTS RECORDED', 'info', 4000);
  if (!winnerShownForCurrentRace) {
    winnerShownForCurrentRace = true;
    setTimeout(() => {
      showWinnerScreen();
    }, 500);
  }
}

function startLiveConnection(url) {
  if (liveSocket) {
    try {
      liveSocket.close();
    } catch (_) {}
    liveSocket = null;
  }
  let previous = {};
  setLiveControls(true);
  showAlert(`CONNECTING TO LIVE STREAM: ${url}`, 'info', 2500);
  liveSocket = connectLive(url, {
    onOpen: () => {
      showAlert('LIVE WEBSOCKET CONNECTED — AWAITING START', 'info', 2500);
    },
    onMessage: (raw) => {
      try {
        const parsed = parseLiveMessage(raw, previous);
        previous = { runId: parsed.runId, seq: parsed.seq };
        if (parsed.kind === 'synapse.live.hello') handleLiveHello(parsed.message);
        else if (parsed.kind === 'synapse.live.state') handleLiveState(parsed.message);
        else if (parsed.kind === 'synapse.live.end') handleLiveEnd(parsed.message);
      } catch (error) {
        showAlert(`LIVE STREAM ERROR: ${error.message}`, 'warn', 3000);
      }
    },
    onError: () => {
      showAlert('LIVE WEBSOCKET CONNECTION ERROR', 'warn', 3000);
    },
    onClose: () => {
      if (!liveEnded && liveMode) {
        showAlert('LIVE CONNECTION CLOSED BY SERVER', 'warn', 3000);
      }
    },
  });
}

/* Runtime state for tracks and screens */
let currentTrackKey = 'oval';
let isAnimating = false;
let winnerShownForCurrentRace = false;
let isHeroMode = true;
if (typeof document !== 'undefined' && document.body) {
  document.body.classList.add('hero-active');
}
let heroShotIndex = 0;
let heroShotTimer = 0;
let customReplayA = null;
let customReplayB = null;

/*
 * 6 Aesthetic F1 Broadcast Highlights and Angles for Hero Screen Showcase
 *
 * I curated 6 dynamic camera perspectives to showcase the cars and circuit in the hero background:
 * 1. Apex Curb Ground Cam: low angle rumble curb perspective clipping corner entry
 * 2. Telephoto Straight Zoom: 24 degree field of view compressed distance head on shot
 * 3. Trackside Lateral Tracking Pan: side by side wheel and halo tracking angle
 * 4. Helicopter Drone Overhead Sweep: elevated wide circuit sweep
 * 5. Low Rear Diffuser and Rain LED Chase: tailing titanium exhaust and flashing LED
 * 6. Spectator Grandstand POV: viewing the cars from the bleachers behind waving flags
 */
const CINEMATIC_SHOTS = [
  {
    id: 'apex_curb',
    name: 'SHOT 1/6 — APEX CURB GROUND CAM',
    fov: 46,
    duration: 4.5,
    update: (posA, posB, rotYA, rotYB, targetPos, targetLook) => {
      const fwdX = Math.sin(rotYA);
      const fwdZ = Math.cos(rotYA);
      const rightX = fwdZ;
      const rightZ = -fwdX;
      targetPos.set(posA.x + fwdX * 5.5 + rightX * 3.4, 0.42, posA.z + fwdZ * 5.5 + rightZ * 3.4);
      targetLook.set(posA.x + fwdX * 1.2, 0.75, posA.z + fwdZ * 1.2);
    }
  },
  {
    id: 'telephoto_straight',
    name: 'SHOT 2/6 — TELEPHOTO STRAIGHT ZOOM',
    fov: 24, /* Extreme telephoto lens compression */
    duration: 4.5,
    update: (posA, posB, rotYA, rotYB, targetPos, targetLook) => {
      const avgRot = lerpAngle(rotYA, rotYB, 0.5);
      const fwdX = Math.sin(avgRot);
      const fwdZ = Math.cos(avgRot);
      const midX = (posA.x + posB.x) * 0.5;
      const midZ = (posA.z + posB.z) * 0.5;
      targetPos.set(midX + fwdX * 36.0, 1.35, midZ + fwdZ * 36.0);
      targetLook.set(midX - fwdX * 2.0, 0.85, midZ - fwdZ * 2.0);
    }
  },
  {
    id: 'side_tracking',
    name: 'SHOT 3/6 — TRACKSIDE LATERAL TRACKING PAN',
    fov: 42,
    duration: 4.5,
    update: (posA, posB, rotYA, rotYB, targetPos, targetLook) => {
      const avgRot = lerpAngle(rotYA, rotYB, 0.5);
      const fwdX = Math.sin(avgRot);
      const fwdZ = Math.cos(avgRot);
      const sideX = -fwdZ;
      const sideZ = fwdX;
      const midX = (posA.x + posB.x) * 0.5;
      const midZ = (posA.z + posB.z) * 0.5;
      targetPos.set(midX + sideX * 9.8 + fwdX * 1.2, 1.65, midZ + sideZ * 9.8 + fwdZ * 1.2);
      targetLook.set(midX + fwdX * 2.8, 1.05, midZ + fwdZ * 2.8);
    }
  },
  {
    id: 'heli_drone',
    name: 'SHOT 4/6 — HELICOPTER DRONE OVERHEAD SWEEP',
    fov: 62,
    duration: 4.5,
    update: (posA, posB, rotYA, rotYB, targetPos, targetLook) => {
      const avgRot = lerpAngle(rotYA, rotYB, 0.5);
      const fwdX = Math.sin(avgRot);
      const fwdZ = Math.cos(avgRot);
      const midX = (posA.x + posB.x) * 0.5;
      const midZ = (posA.z + posB.z) * 0.5;
      targetPos.set(midX - fwdX * 24.0, 18.5, midZ - fwdZ * 24.0);
      targetLook.set(midX + fwdX * 8.0, 0.5, midZ + fwdZ * 8.0);
    }
  },
  {
    id: 'rear_diffuser',
    name: 'SHOT 5/6 — LOW REAR DIFFUSER & RAIN LED CHASE',
    fov: 52,
    duration: 4.5,
    update: (posA, posB, rotYA, rotYB, targetPos, targetLook) => {
      const fwdX = Math.sin(rotYB);
      const fwdZ = Math.cos(rotYB);
      targetPos.set(posB.x - fwdX * 6.2, posB.y + 1.35, posB.z - fwdZ * 6.2);
      targetLook.set(posB.x + fwdX * 7.5, posB.y + 0.9, posB.z + fwdZ * 7.5);
    }
  },
  {
    id: 'grandstand_pov',
    name: 'SHOT 6/6 — SPECTATOR GRANDSTAND POV',
    fov: 44,
    duration: 4.5,
    update: (posA, posB, rotYA, rotYB, targetPos, targetLook) => {
      const avgRot = lerpAngle(rotYA, rotYB, 0.5);
      const fwdX = Math.sin(avgRot);
      const fwdZ = Math.cos(avgRot);
      const sideX = fwdZ;
      const sideZ = -fwdX;
      const midX = (posA.x + posB.x) * 0.5;
      const midZ = (posA.z + posB.z) * 0.5;
      targetPos.set(midX + sideX * 16.0 - fwdX * 6.0, 7.2, midZ + sideZ * 16.0 - fwdZ * 6.0);
      targetLook.set(midX + fwdX * 4.0, 1.0, midZ + fwdZ * 4.0);
    }
  }
];

/*
 * Here I manage the repeating 6 cinematic highlight shots and optical broadcast cuts in Hero Mode.
 * I cycle through dynamic camera perspectives with optical shutter flashes to showcase the racing action.
 */
function updateHeroCinematicCamera(posA, posB, rotYA, rotYB, dt) {
  heroShotTimer += dt;
  const currentShot = CINEMATIC_SHOTS[heroShotIndex];

  if (heroShotTimer >= currentShot.duration) {
    heroShotTimer = 0;
    heroShotIndex = (heroShotIndex + 1) % CINEMATIC_SHOTS.length;
    const nextShot = CINEMATIC_SHOTS[heroShotIndex];

    if (heroCamLabel) {
      heroCamLabel.textContent = `CINEMATIC HIGHLIGHTS: ${nextShot.name}`;
    }

    /* Trigger optical broadcast shutter wipe flash */
    if (cutFlashElem) {
      cutFlashElem.classList.remove('active');
      void cutFlashElem.offsetWidth; /* Force CSS animation reflow */
      cutFlashElem.classList.add('active');
      setTimeout(() => cutFlashElem?.classList.remove('active'), 200);
    }

    /* Switch camera lens field of view */
    if (camera.fov !== nextShot.fov) {
      camera.fov = nextShot.fov;
      camera.updateProjectionMatrix();
    }

    /* Snap to new camera cut position instantly without disorienting traversal interpolation */
    nextShot.update(posA, posB, rotYA, rotYB, camTargetPos, camTargetLook);
    camCurrentPos.copy(camTargetPos);
    camCurrentLook.copy(camTargetLook);
    camera.position.copy(camCurrentPos);
    camera.lookAt(camCurrentLook);
    return;
  }

  /* Smooth dynamic tracking within the current camera cut */
  currentShot.update(posA, posB, rotYA, rotYB, camTargetPos, camTargetLook);
  const SHOT_SMOOTH = 1 - Math.pow(0.005, dt);
  camCurrentPos.lerp(camTargetPos, SHOT_SMOOTH);
  camCurrentLook.lerp(camTargetLook, SHOT_SMOOTH);
  camera.position.copy(camCurrentPos);
  camera.lookAt(camCurrentLook);
}

/* Show a broadcast banner message for overtakes, safety flags, and race alerts */
function showAlert(text, type = 'info', durationMs = 2500) {
  if (!broadcastAlert) return;
  alertText.textContent = text;
  broadcastAlert.className = 'broadcast-alert ' + (type === 'warn' ? 'yellow' : type === 'danger' ? 'red' : '');
  broadcastAlert.classList.remove('hidden');
  clearTimeout(showAlert._timer);
  showAlert._timer = setTimeout(() => {
    broadcastAlert.classList.add('hidden');
  }, durationMs);
}

/* Format seconds as minutes, seconds, and milliseconds */
function formatTime(s) {
  const ms = Math.max(0, Math.round(s * 1000));
  const min = Math.floor(ms / 60000);
  const sec = Math.floor((ms % 60000) / 1000);
  const mil = ms % 1000;
  return `${String(min).padStart(2, '0')}:${String(sec).padStart(2, '0')}.${String(mil).padStart(3, '0')}`;
}

/*
 * I update the Formula 1 steering wheel 15 LED shift light bar.
 * My approach calculates the active gear ratio and fraction towards redline:
 * LEDs 0 to 4 illuminate in green, LEDs 5 to 9 illuminate in red, and LEDs 10 to 14 flash blue at shift point.
 */
function updateRPMShiftLights(elemId, gear, speedKmh) {
  const bar = document.getElementById(elemId);
  if (!bar) return;
  const leds = bar.querySelectorAll('.rpm-led');
  if (!leds || leds.length !== 15) return;

  const gearSpeeds = [0, 75, 120, 165, 210, 255, 295, 335, 370];
  const gMin = gearSpeeds[gear - 1] || 0;
  const gMax = gearSpeeds[gear] || 370;
  const frac = Math.max(0, Math.min(1, (speedKmh - gMin) / (gMax - gMin + 0.1)));

  /* 15 LEDs: 0 to 4 green, 5 to 9 red, 10 to 14 blue */
  const litCount = Math.floor(frac * 16);
  const isRedline = frac > 0.92;

  leds.forEach((led, idx) => {
    led.classList.toggle('active', idx < litCount);
    if (idx >= 10) {
      led.classList.toggle('flash', isRedline);
    } else {
      led.classList.remove('flash');
    }
  });
}

/*
 * Here I initialize the Three.js WebGL renderer, perspective camera, tone mapping, and stadium night lighting.
 * I configure deep cyberpunk midnight sky, atmospheric night haze, soft shadow maps, and ACES Filmic tone mapping.
 */
function initThreeJS() {
  scene = new THREE.Scene();

  /* Cyberpunk midnight atmosphere */
  scene.background = new THREE.Color(0x060c18);
  scene.fog = new THREE.FogExp2(0x070e1c, 0.00065);

  camera = new THREE.PerspectiveCamera(54, window.innerWidth / window.innerHeight, 0.4, 4500);
  camera.position.set(0, 60, 0);

  renderer3d = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, powerPreference: 'high-performance' });
  renderer3d.setSize(window.innerWidth, window.innerHeight);
  renderer3d.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer3d.shadowMap.enabled = true;
  renderer3d.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer3d.toneMapping = THREE.ACESFilmicToneMapping;
  renderer3d.toneMappingExposure = 1.25;

  /* Directional moonlight providing crisp highlights across the track and car livery */
  mainSunLight = new THREE.DirectionalLight(0x88b0e8, 1.85);
  mainSunLight.position.set(70, 190, 50);
  mainSunLight.castShadow = true;
  mainSunLight.shadow.mapSize.width = 2048;
  mainSunLight.shadow.mapSize.height = 2048;
  mainSunLight.shadow.camera.near = 10;
  mainSunLight.shadow.camera.far = 1200;
  mainSunLight.shadow.camera.left = -300;
  mainSunLight.shadow.camera.right = 300;
  mainSunLight.shadow.camera.top = 300;
  mainSunLight.shadow.camera.bottom = -300;
  mainSunLight.shadow.bias = -0.0003;
  scene.add(mainSunLight);
  scene.add(mainSunLight.target);

  /* Deep sky and turf hemisphere lighting */
  hemiLight = new THREE.HemisphereLight(0x384f73, 0x121e16, 1.15);
  scene.add(hemiLight);

  /* Soft ambient night fill ensuring grandstands and outfield are lightly visible */
  ambientLight = new THREE.AmbientLight(0x283852, 1.25);
  scene.add(ambientLight);

  /* Vast night stadium outfield ground plane */
  const groundGeo = new THREE.PlaneGeometry(6000, 6000);
  const groundMat = new THREE.MeshStandardMaterial({
    color: 0x12241b,
    roughness: 0.92,
    metalness: 0.02,
  });
  groundMesh = new THREE.Mesh(groundGeo, groundMat);
  groundMesh.rotation.x = -Math.PI / 2;
  groundMesh.position.y = -0.06;
  groundMesh.receiveShadow = true;
  scene.add(groundMesh);

  clock = new THREE.Clock();
  window.addEventListener('resize', onResize);
}

function onResize() {
  if (!camera || !renderer3d) return;
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer3d.setSize(window.innerWidth, window.innerHeight);
}

/*
 * I load the selected circuit and both racing cars into the Three.js scene,
 * configuring cameras and initial starting grid placement.
 */
function loadRaceIntoScene() {
  const track = replayA.track;

  /* Reset spectator flags */
  spectatorFlags = [];

  /* Build track environment */
  trackGroup = buildTrackScene(track);
  scene.add(trackGroup);

  /* Compute track bounds for top down overhead camera */
  const allPoints = [...track.barrier_left, ...track.barrier_right];
  trackBounds.minX = Infinity; trackBounds.maxX = -Infinity;
  trackBounds.minZ = Infinity; trackBounds.maxZ = -Infinity;
  for (const p of allPoints) {
    trackBounds.minX = Math.min(trackBounds.minX, p[0]);
    trackBounds.maxX = Math.max(trackBounds.maxX, p[0]);
    trackBounds.minZ = Math.min(trackBounds.minZ, -p[1]);
    trackBounds.maxZ = Math.max(trackBounds.maxZ, -p[1]);
  }

  /* Dynamic scene scaling for any circuit scale */
  const spanX = trackBounds.maxX - trackBounds.minX;
  const spanZ = trackBounds.maxZ - trackBounds.minZ;
  const maxDim = Math.max(spanX, spanZ, 450);
  const centerX = (trackBounds.minX + trackBounds.maxX) / 2;
  const centerZ = (trackBounds.minZ + trackBounds.maxZ) / 2;

  if (camera) {
    camera.far = Math.max(4500, maxDim * 4.5);
    camera.updateProjectionMatrix();
  }

  if (groundMesh) {
    groundMesh.position.set(centerX, -0.06, centerZ);
    const reqGroundSize = Math.max(8000, maxDim * 3.8);
    groundMesh.scale.set(reqGroundSize / 6000, reqGroundSize / 6000, 1);
  }

  if (mainSunLight) {
    mainSunLight.position.set(centerX + maxDim * 0.45, maxDim * 0.85, centerZ + maxDim * 0.35);
    mainSunLight.target.position.set(centerX, 0, centerZ);
    mainSunLight.target.updateMatrixWorld();
    mainSunLight.shadow.camera.near = 10;
    mainSunLight.shadow.camera.far = maxDim * 2.8;
    mainSunLight.shadow.camera.left = -maxDim * 0.65;
    mainSunLight.shadow.camera.right = maxDim * 0.65;
    mainSunLight.shadow.camera.top = maxDim * 0.65;
    mainSunLight.shadow.camera.bottom = -maxDim * 0.65;
    mainSunLight.shadow.camera.updateProjectionMatrix();
  }

  if (scene && scene.fog) {
    scene.fog.density = 0.65 / (maxDim * 3.5);
  }

  /* Build F1 Cars:
   Car A: Electric Cyan with pure white accents
   Car B: Radiant Papaya Solar Orange with solar gold accents */
  carModelA = buildF1CarModel(0x00f0ff, 0xffffff, '1');
  carModelB = buildF1CarModel(0xff6600, 0xffd166, '2');
  scene.add(carModelA);
  scene.add(carModelB);

  /* Determine race duration */
  const durA = replayA.frames.length > 1 ? (replayA.frames.length - 1) / replayA.tickHz : 0;
  const durB = replayB.frames.length > 1 ? (replayB.frames.length - 1) / replayB.tickHz : 0;
  if (!liveMode) maxTime = Math.max(durA, durB);

  /* Update HUD text */
  document.getElementById('p1-name').textContent = replayA.participantId.toUpperCase();
  document.getElementById('p2-name').textContent = replayB.participantId.toUpperCase();
  document.getElementById('telem-name-a').textContent = replayA.participantId.toUpperCase();
  document.getElementById('telem-name-b').textContent = replayB.participantId.toUpperCase();
  timeTotal.textContent = liveMode ? 'LIVE' : formatTime(maxTime);

  const trackId = replayA.track.meta?.id || replayA.track.meta?.track_id || 'CIRCUIT';
  const trackTag = document.getElementById('track-name-tag');
  if (trackTag) {
    if (trackId.includes('monaco')) trackTag.textContent = 'MONACO GP';
    else if (trackId.includes('spa')) trackTag.textContent = 'SPA-FRANCORCHAMPS';
    else trackTag.textContent = trackId.replace('synthetic_', '').toUpperCase();
  }

  if (trackId.includes('flowing')) currentTrackKey = 'flowing';
  else if (trackId.includes('tight')) currentTrackKey = 'tight';
  else if (trackId.includes('monaco')) currentTrackKey = 'monaco';
  else if (trackId.includes('spa')) currentTrackKey = 'spa';
  else if (trackId.includes('oval')) currentTrackKey = 'oval';
  if (trackSwitcher) trackSwitcher.value = currentTrackKey;

  /* Initial camera position behind start line */
  const f0 = replayA.frames[0];
  const initAngle = Math.atan2(Math.cos(f0.heading), -Math.sin(f0.heading));
  const fwdX = Math.sin(initAngle);
  const fwdZ = Math.cos(initAngle);

  camCurrentPos.set(f0.x - fwdX * 18, 7.5, -f0.y - fwdZ * 18);
  camCurrentLook.set(f0.x, 1.2, -f0.y);
  camera.position.copy(camCurrentPos);
  camera.lookAt(camCurrentLook);

  showAlert('GREEN FLAG — RACE STARTED', 'info', 3000);
}

/* Vehicle Positioning, Orientation, and Telemetry Simulation */

/*
 * Master vehicle telemetry update loop.
 * Here I interpolate car coordinates, calculate heading angles, animate steering and wheels,
 * compute dynamic overlap transparency, flare glowing brake discs, and update live timing.
 */
function updateCars(timeS, dt) {
  const fallbackA = sampleAtTime(replayA.frames, replayA.tickHz, timeS);
  const fallbackB = sampleAtTime(replayB.frames, replayB.tickHz, timeS);
  const sA = liveMode ? (liveCars[0] || fallbackA) : fallbackA;
  const sB = liveMode ? (liveCars[1] || fallbackB) : fallbackB;

  /* 1. Calculate continuous vehicle yaw heading angles using shortest arc interpolation */
  const rotYA = Math.atan2(Math.cos(sA.heading), -Math.sin(sA.heading));
  const rotYB = Math.atan2(Math.cos(sB.heading), -Math.sin(sB.heading));

  /* 2. Compute lateral lane offsets so side by side racing avoids visual collision */
  let offAx = 0, offAz = 0;
  let offBx = 0, offBz = 0;

  if (laneSeparation) {
    const sepM = 1.35;
    offAx = -Math.cos(rotYA) * sepM;
    offAz = Math.sin(rotYA) * sepM;

    offBx = Math.cos(rotYB) * sepM;
    offBz = -Math.sin(rotYB) * sepM;
  }

  /* Position root car groups */
  carModelA.position.set(sA.x + offAx, 0, -sA.y + offAz);
  carModelA.rotation.y = rotYA;

  carModelB.position.set(sB.x + offBx, 0, -sB.y + offBz);
  carModelB.rotation.y = rotYB;

  /* 3. Dynamic overlap transparency and glowing wireframe matrix */
  const carDist = carModelA.position.distanceTo(carModelB.position);
  const OVERLAP_THRESHOLD = 2.8;
  const overlapFactor = Math.max(0, Math.min(1, (OVERLAP_THRESHOLD - carDist) / 2.2));

  const targetBodyOpacity = 1.0 - overlapFactor * 0.55;
  const targetWireOpacity = overlapFactor * 0.95;

  for (const m of carModelA.userData.bodyMaterials) { m.opacity = targetBodyOpacity; }
  for (const m of carModelA.userData.edgeMaterials) { m.opacity = targetWireOpacity; }
  for (const m of carModelA.userData.wireMaterials) { m.opacity = targetWireOpacity * 0.35; }

  for (const m of carModelB.userData.bodyMaterials) { m.opacity = targetBodyOpacity; }
  for (const m of carModelB.userData.edgeMaterials) { m.opacity = targetWireOpacity; }
  for (const m of carModelB.userData.wireMaterials) { m.opacity = targetWireOpacity * 0.35; }

  if (overlapIndicator) {
    overlapIndicator.classList.toggle('hidden', overlapFactor <= 0.25);
  }

  /* 4. Update wheel spin, steering lock, chassis roll, and brake disc heating */
  const wheelRadius = 0.34;

  for (const w of carModelA.userData.frontWheels) {
    w.rotation.y = sA.steer * 0.7;
  }
  for (const w of carModelA.userData.wheels) {
    w.rotation.x += (sA.speed / wheelRadius) * dt;
  }

  for (const w of carModelB.userData.frontWheels) {
    w.rotation.y = sB.steer * 0.7;
  }
  for (const w of carModelB.userData.wheels) {
    w.rotation.x += (sB.speed / wheelRadius) * dt;
  }

  /* Glowing carbon brake discs: emission glows hot red and orange during heavy deceleration */
  const thrA = Math.max(0, sA.throttle_brake);
  const brkA = Math.max(0, -sA.throttle_brake);
  if (carModelA.userData.brakeDiscs) {
    const bIntA = brkA > 0.12 ? Math.min(1, brkA * 1.5) : 0;
    for (const d of carModelA.userData.brakeDiscs) {
      d.emissive.setHex(0xff2200);
      d.emissiveIntensity = bIntA * 3.0;
    }
  }

  const thrB = Math.max(0, sB.throttle_brake);
  const brkB = Math.max(0, -sB.throttle_brake);
  if (carModelB.userData.brakeDiscs) {
    const bIntB = brkB > 0.12 ? Math.min(1, brkB * 1.5) : 0;
    for (const d of carModelB.userData.brakeDiscs) {
      d.emissive.setHex(0xff2200);
      d.emissiveIntensity = bIntB * 3.0;
    }
  }

  const chassisA = carModelA.userData.chassis;
  chassisA.rotation.z = -sA.yaw_rate * 0.035;
  chassisA.rotation.x = sA.throttle_brake < 0 ? sA.throttle_brake * 0.025 : 0;

  const chassisB = carModelB.userData.chassis;
  chassisB.rotation.z = -sB.yaw_rate * 0.035;
  chassisB.rotation.x = sB.throttle_brake < 0 ? sB.throttle_brake * 0.025 : 0;

  /* Pulsing rear rain safety LED and glowing molten titanium exhaust */
  const pulseIntensity = (Math.sin(timeS * 14) + 1) * 0.5;
  if (carModelA.userData.rainLed) {
    carModelA.userData.rainLed.material.color.setRGB(0.5 + pulseIntensity * 0.5, 0, 0.05);
    if (carModelA.userData.rainLight) carModelA.userData.rainLight.intensity = 1.0 + pulseIntensity * 2.2;
  }
  if (carModelA.userData.exhaustCore) {
    const boostA = Math.max(0, sA.throttle_brake);
    carModelA.userData.exhaustCore.material.color.setRGB(1.0, 0.4 + boostA * 0.5, boostA * 0.2);
    if (carModelA.userData.exhaustGlow) carModelA.userData.exhaustGlow.intensity = 0.4 + boostA * 2.2 + Math.random() * 0.3;
  }

  if (carModelB.userData.rainLed) {
    carModelB.userData.rainLed.material.color.setRGB(0.5 + pulseIntensity * 0.5, 0, 0.05);
    if (carModelB.userData.rainLight) carModelB.userData.rainLight.intensity = 1.0 + pulseIntensity * 2.2;
  }
  if (carModelB.userData.exhaustCore) {
    const boostB = Math.max(0, sB.throttle_brake);
    carModelB.userData.exhaustCore.material.color.setRGB(1.0, 0.4 + boostB * 0.5, boostB * 0.2);
    if (carModelB.userData.exhaustGlow) carModelB.userData.exhaustGlow.intensity = 0.4 + boostB * 2.2 + Math.random() * 0.3;
  }

  /* 5. Update driver telemetry HUD gauges, shift lights, and DRS status */
  const speedKmhA = Math.round(sA.speed * 3.6);
  const speedKmhB = Math.round(sB.speed * 3.6);
  document.getElementById('speed-a').textContent = speedKmhA;
  document.getElementById('speed-b').textContent = speedKmhB;

  const gA = Math.abs((sA.speed * sA.yaw_rate) / 9.81);
  const gB = Math.abs((sB.speed * sB.yaw_rate) / 9.81);
  const gElemA = document.getElementById('gforce-a');
  const gElemB = document.getElementById('gforce-b');
  if (gElemA) gElemA.textContent = `${gA.toFixed(1)} G`;
  if (gElemB) gElemB.textContent = `${gB.toFixed(1)} G`;

  const degA = (sA.steer * (180 / Math.PI)).toFixed(1);
  const degB = (sB.steer * (180 / Math.PI)).toFixed(1);
  const steerValA = document.getElementById('steer-val-a');
  const steerValB = document.getElementById('steer-val-b');
  if (steerValA) steerValA.textContent = `${sA.steer >= 0 ? '+' : ''}${degA}°`;
  if (steerValB) steerValB.textContent = `${sB.steer >= 0 ? '+' : ''}${degB}°`;

  const gearA = Math.min(8, Math.max(1, Math.floor(speedKmhA / 38) + 1));
  const gearB = Math.min(8, Math.max(1, Math.floor(speedKmhB / 38) + 1));
  document.getElementById('gear-a').textContent = gearA;
  document.getElementById('gear-b').textContent = gearB;

  /* Steering wheel 15 LED RPM shift light bar */
  updateRPMShiftLights('rpm-bar-a', gearA, speedKmhA);
  updateRPMShiftLights('rpm-bar-b', gearB, speedKmhB);

  /* DRS activation indicator */
  const drsA = document.getElementById('drs-a');
  if (drsA) drsA.classList.toggle('active', thrA > 0.92 && speedKmhA > 150);
  const drsB = document.getElementById('drs-b');
  if (drsB) drsB.classList.toggle('active', thrB > 0.92 && speedKmhB > 150);

  document.getElementById('throttle-a').style.width = `${Math.min(100, thrA * 100)}%`;
  document.getElementById('brake-a').style.width = `${Math.min(100, brkA * 100)}%`;
  document.getElementById('steer-a').style.left = `${Math.max(5, Math.min(95, 50 + (sA.steer / 0.55) * 45))}%`;

  document.getElementById('throttle-b').style.width = `${Math.min(100, thrB * 100)}%`;
  document.getElementById('brake-b').style.width = `${Math.min(100, brkB * 100)}%`;
  document.getElementById('steer-b').style.left = `${Math.max(5, Math.min(95, 50 + (sB.steer / 0.55) * 45))}%`;

  /* 6. Update F1 live timing tower, interval splits, and incident notifications */
  const tickA = sA.tick;
  const tickB = sB.tick;

  let leaderIsA = true;
  let gapStr = 'LEADER';

  if (sA.finished && !sB.finished) {
    leaderIsA = true;
    gapStr = replayB.result.status?.replace('dnf_', 'DNF ').toUpperCase() || '+1 LAP';
  } else if (sB.finished && !sA.finished) {
    leaderIsA = false;
    gapStr = replayA.result.status?.replace('dnf_', 'DNF ').toUpperCase() || '+1 LAP';
  } else if (tickA >= tickB) {
    leaderIsA = true;
    const tickDiff = tickA - tickB;
    gapStr = tickDiff > 0 ? `+${(tickDiff / replayA.tickHz).toFixed(2)}s` : 'INTERVAL 0.00s';
  } else {
    leaderIsA = false;
    const tickDiff = tickB - tickA;
    gapStr = tickDiff > 0 ? `+${(tickDiff / replayB.tickHz).toFixed(2)}s` : 'INTERVAL 0.00s';
  }

  /* Overtake detection alert */
  const currentLeaderName = leaderIsA ? replayA.participantId.toUpperCase() : replayB.participantId.toUpperCase();
  if (lastLeader !== null && lastLeader !== currentLeaderName && timeS > 1.0) {
    showAlert(`⚡ OVERTAKE — ${currentLeaderName} TAKES P1!`, 'info', 3000);
  }
  lastLeader = currentLeaderName;

  /* Off track track limits warnings */
  if (sA.off_track && !lastOffTrackA) {
    showAlert(`⚠️ TRACK LIMITS — ${replayA.participantId.toUpperCase()} OFF TRACK`, 'warn', 2200);
  }
  lastOffTrackA = sA.off_track;

  if (sB.off_track && !lastOffTrackB) {
    showAlert(`⚠️ TRACK LIMITS — ${replayB.participantId.toUpperCase()} OFF TRACK`, 'warn', 2200);
  }
  lastOffTrackB = sB.off_track;

  /* Sector calculation for sectors 1, 2, and 3 */
  const progressFraction = Math.max(0, Math.min(1, maxTime > 0 ? timeS / maxTime : 0));
  const sector = progressFraction < 0.33 ? 'SEC 1' : progressFraction < 0.66 ? 'SEC 2' : 'SEC 3';
  if (towerSector) towerSector.textContent = sector;

  /* Mini sector status chips in timing tower */
  const s1a = document.getElementById('p1-s1');
  const s2a = document.getElementById('p1-s2');
  const s3a = document.getElementById('p1-s3');
  if (s1a && s2a && s3a) {
    s1a.className = 'sec-dot ' + (progressFraction >= 0.33 ? 'purple' : 'yellow');
    s2a.className = 'sec-dot ' + (progressFraction >= 0.66 ? 'green' : progressFraction >= 0.33 ? 'yellow' : '');
    s3a.className = 'sec-dot ' + (progressFraction >= 0.95 ? 'purple' : progressFraction >= 0.66 ? 'yellow' : '');
  }

  const p1NameElem = document.getElementById('p1-name');
  const p2NameElem = document.getElementById('p2-name');
  const p1GapElem = document.getElementById('p1-gap');
  const p2GapElem = document.getElementById('p2-gap');
  const p1Ind = document.getElementById('p1-indicator');
  const p2Ind = document.getElementById('p2-indicator');

  if (leaderIsA) {
    p1NameElem.textContent = replayA.participantId.toUpperCase();
    p1GapElem.textContent = 'LEADER';
    p1Ind.className = 'driver-indicator cyan-bg';

    p2NameElem.textContent = replayB.participantId.toUpperCase();
    p2GapElem.textContent = gapStr;
    p2Ind.className = 'driver-indicator gold-bg';
  } else {
    p1NameElem.textContent = replayB.participantId.toUpperCase();
    p1GapElem.textContent = 'LEADER';
    p1Ind.className = 'driver-indicator gold-bg';

    p2NameElem.textContent = replayA.participantId.toUpperCase();
    p2GapElem.textContent = gapStr;
    p2Ind.className = 'driver-indicator cyan-bg';
  }

  if (progressBar) progressBar.style.width = `${progressFraction * 100}%`;

  return { sA, sB, rotYA, rotYB };
}

/* Phase 4: Dynamic Chase Camera System and Broadcast Tracking */

/* I update the broadcast chase camera position with smooth exponential damping and lookahead */
function updateCamera(sA, sB, rotYA, rotYB, dt) {
  const posA = carModelA.position;
  const posB = carModelB.position;

  /* In hero mode, execute broadcast highlight cuts across 6 cinematic angles */
  if (isHeroMode) {
    updateHeroCinematicCamera(posA, posB, rotYA, rotYB, dt);
    return;
  }

  const SMOOTH = 1 - Math.pow(0.0008, dt);

  if (cameraMode === 'top') {
    const cx = (trackBounds.minX + trackBounds.maxX) / 2;
    const cz = (trackBounds.minZ + trackBounds.maxZ) / 2;
    const span = Math.max(trackBounds.maxX - trackBounds.minX, trackBounds.maxZ - trackBounds.minZ);
    camTargetPos.set(cx, span * 0.88, cz);
    camTargetLook.set(cx, 0, cz);
  } else if (cameraMode === 'cockpit') {
    const fwdX = Math.sin(rotYA);
    const fwdZ = Math.cos(rotYA);
    camTargetPos.set(posA.x + fwdX * 0.8, posA.y + 0.85, posA.z + fwdZ * 0.8);
    camTargetLook.set(posA.x + fwdX * 30, posA.y + 0.7, posA.z + fwdZ * 30);
  } else if (cameraMode === 'carA') {
    const fwdX = Math.sin(rotYA);
    const fwdZ = Math.cos(rotYA);
    const dist = 13.5;
    const height = 5.6;

    camTargetPos.set(posA.x - fwdX * dist, posA.y + height, posA.z - fwdZ * dist);
    camTargetLook.set(posA.x + fwdX * 4, posA.y + 1.2, posA.z + fwdZ * 4);
  } else if (cameraMode === 'carB') {
    const fwdX = Math.sin(rotYB);
    const fwdZ = Math.cos(rotYB);
    const dist = 13.5;
    const height = 5.6;

    camTargetPos.set(posB.x - fwdX * dist, posB.y + height, posB.z - fwdZ * dist);
    camTargetLook.set(posB.x + fwdX * 4, posB.y + 1.2, posB.z + fwdZ * 4);
  } else if (cameraMode === 'side') {
    /* Dynamic trackside TV side profile camera angle */
    const midX = (posA.x + posB.x) * 0.5;
    const midZ = (posA.z + posB.z) * 0.5;
    const avgRotY = lerpAngle(rotYA, rotYB, 0.5);
    const fwdX = Math.sin(avgRotY);
    const fwdZ = Math.cos(avgRotY);
    /* Perpendicular side vector observing the cars from trackside */
    const sideX = -fwdZ;
    const sideZ = fwdX;
    const sideDist = 11.5;
    const fwdDist = 2.2;
    const camHeight = 2.3; /* Low angle dynamic TV camera height */

    camTargetPos.set(midX + sideX * sideDist + fwdX * fwdDist, camHeight, midZ + sideZ * sideDist + fwdZ * fwdDist);
    camTargetLook.set(midX + fwdX * 2.2, 1.1, midZ + fwdZ * 2.2);
  } else {
    /* Default battle chase camera */
    const midX = (posA.x + posB.x) * 0.5;
    const midZ = (posA.z + posB.z) * 0.5;
    const carDist = posA.distanceTo(posB);

    const avgRotY = lerpAngle(rotYA, rotYB, 0.5);
    const fwdX = Math.sin(avgRotY);
    const fwdZ = Math.cos(avgRotY);

    const dist = Math.min(32, 17 + carDist * 0.55);
    const height = Math.min(14, 6.5 + carDist * 0.25);

    camTargetPos.set(midX - fwdX * dist, height, midZ - fwdZ * dist);
    camTargetLook.set(midX + fwdX * 5, 1.2, midZ + fwdZ * 5);
  }

  camCurrentPos.lerp(camTargetPos, SMOOTH);
  camCurrentLook.lerp(camTargetLook, SMOOTH);

  camera.position.copy(camCurrentPos);
  camera.lookAt(camCurrentLook);
}

/* Phase 5: Radar Minimap with Live Camera View Frustum Cone */

/* I render the 2D circuit radar minimap on the overlay canvas */
function drawMinimap(sA, sB, rotYA, rotYB) {
  const mc = document.getElementById('minimap-canvas');
  if (!mc) return;
  const ctx = mc.getContext('2d');
  const w = mc.width;
  const h = mc.height;

  ctx.clearRect(0, 0, w, h);

  const pad = 18;
  const spanX = Math.max(1, trackBounds.maxX - trackBounds.minX);
  const spanZ = Math.max(1, trackBounds.maxZ - trackBounds.minZ);
  const scale = Math.min((w - pad * 2) / spanX, (h - pad * 2) / spanZ);
  const cx = (trackBounds.minX + trackBounds.maxX) / 2;
  const cz = (trackBounds.minZ + trackBounds.maxZ) / 2;

  function toMap(x3d, z3d) {
    return {
      x: w / 2 + (x3d - cx) * scale,
      y: h / 2 + (z3d - cz) * scale,
    };
  }

  /* 1. Draw Track Surface Ribbon */
  const left = replayA.track.left_boundary;
  const right = replayA.track.right_boundary;
  const n = left.length;

  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const p = toMap(left[i][0], -left[i][1]);
    if (i === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  }
  for (let i = n - 1; i >= 0; i--) {
    const p = toMap(right[i][0], -right[i][1]);
    ctx.lineTo(p.x, p.y);
  }
  ctx.closePath();
  ctx.fillStyle = 'rgba(255, 255, 255, 0.12)';
  ctx.fill();

  /* 2. Draw Track Boundaries */
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const p = toMap(left[i][0], -left[i][1]);
    if (i === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  }
  ctx.closePath();
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.45)';
  ctx.lineWidth = 1.4;
  ctx.stroke();

  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const p = toMap(right[i][0], -right[i][1]);
    if (i === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  }
  ctx.closePath();
  ctx.stroke();

  /* 3. Draw Camera View Frustum Cone */
  const camPosMap = toMap(camera.position.x, camera.position.z);
  const lookDirX = camCurrentLook.x - camera.position.x;
  const lookDirZ = camCurrentLook.z - camera.position.z;
  const camAngle = Math.atan2(lookDirZ, lookDirX);

  const coneLen = 32;
  const coneFov = 0.45;
  ctx.beginPath();
  ctx.moveTo(camPosMap.x, camPosMap.y);
  ctx.lineTo(camPosMap.x + Math.cos(camAngle - coneFov) * coneLen, camPosMap.y + Math.sin(camAngle - coneFov) * coneLen);
  ctx.lineTo(camPosMap.x + Math.cos(camAngle + coneFov) * coneLen, camPosMap.y + Math.sin(camAngle + coneFov) * coneLen);
  ctx.closePath();
  ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
  ctx.fill();

  /* 4. Car A marker in cyan */
  const posA = toMap(carModelA.position.x, carModelA.position.z);
  ctx.save();
  ctx.translate(posA.x, posA.y);
  ctx.rotate(rotYA);

  ctx.beginPath();
  ctx.arc(0, 0, 5, 0, Math.PI * 2);
  ctx.fillStyle = '#00f0ff';
  ctx.shadowColor = '#00f0ff';
  ctx.shadowBlur = 10;
  ctx.fill();
  ctx.shadowBlur = 0;

  ctx.beginPath();
  ctx.moveTo(0, 6);
  ctx.lineTo(2.5, -2);
  ctx.lineTo(-2.5, -2);
  ctx.closePath();
  ctx.fillStyle = '#ffffff';
  ctx.fill();
  ctx.restore();

  /* 5. Car B marker in papaya */
  const posB = toMap(carModelB.position.x, carModelB.position.z);
  ctx.save();
  ctx.translate(posB.x, posB.y);
  ctx.rotate(rotYB);

  ctx.beginPath();
  ctx.arc(0, 0, 5, 0, Math.PI * 2);
  ctx.fillStyle = '#ff8000';
  ctx.shadowColor = '#ff8000';
  ctx.shadowBlur = 10;
  ctx.fill();
  ctx.shadowBlur = 0;

  ctx.beginPath();
  ctx.moveTo(0, 6);
  ctx.lineTo(2.5, -2);
  ctx.lineTo(-2.5, -2);
  ctx.closePath();
  ctx.fillStyle = '#ffffff';
  ctx.fill();
  ctx.restore();
}

/* Animation Render Loop */

function animate() {
  requestAnimationFrame(animate);

  const dt = clock.getDelta();

  if (isHeroMode) {
    /* In hero showcase mode, keep racing simulation looping continuously in background */
    currentTime += dt * playbackSpeed;
    if (maxTime > 0 && currentTime >= maxTime) {
      currentTime = 0; /* Seamless continuous racing highlights loop */
    }
  } else if (!liveMode && playing && maxTime > 0) {
    currentTime += dt * playbackSpeed;
    if (currentTime >= maxTime) {
      currentTime = maxTime;
      playing = false;
      btnPlay.textContent = '▶';
      showAlert('CHECKERED FLAG — RACE FINISHED', 'info', 4000);
      if (!winnerShownForCurrentRace) {
        winnerShownForCurrentRace = true;
        setTimeout(() => {
          showWinnerScreen();
        }, 400);
      }
    }
  }

  if (liveMode && !liveEnded && liveLastPacketAt > 0) {
    const staleAfterMs = Math.max(250, (3 / livePhysicsHz) * 1000);
    if (performance.now() - liveLastPacketAt > staleAfterMs && !liveStaleShown) {
      liveStaleShown = true;
      if (towerSession) towerSession.textContent = 'STALE';
      showAlert('LIVE STREAM STALE — HOLDING LAST AUTHORITATIVE STATE', 'warn', 2500);
    }
  }

  if (!liveMode && scrubber && maxTime > 0) {
    scrubber.value = (currentTime / maxTime) * 1000;
  }
  if (timeCurrent) timeCurrent.textContent = formatTime(currentTime);

  const { sA, sB, rotYA, rotYB } = updateCars(currentTime, dt);
  updateCamera(sA, sB, rotYA, rotYB, dt);
  drawMinimap(sA, sB, rotYA, rotYB);

  /* Synchronize audio engine with vehicle speeds, camera focus, and hero mode state */
  soundSystem.update(sA, sB, playing, cameraMode, isHeroMode);

  renderer3d.render(scene, camera);
}

/* Winner Victory Podium Display and Circuit Switcher */

/*
 * Official provisional classification winner podium modal.
 * When the winning car crosses the finish line, I show this podium modal with final gap times,
 * lap records, and options to replay or rotate to the next circuit.
 */
function showWinnerScreen() {
  if (!winnerModal || !replayA || !replayB) return;

  const aFinished = replayA.result?.status === 'finished';
  const bFinished = replayB.result?.status === 'finished';

  const durA = replayA.frames.length > 1 ? (replayA.frames.length - 1) / replayA.tickHz : 0;
  const durB = replayB.frames.length > 1 ? (replayB.frames.length - 1) / replayB.tickHz : 0;

  const timeA = replayA.result?.elapsed_s || durA;
  const timeB = replayB.result?.elapsed_s || durB;

  let aWins = true;
  if (aFinished && !bFinished) {
    aWins = true;
  } else if (!aFinished && bFinished) {
    aWins = false;
  } else if (aFinished && bFinished) {
    aWins = timeA <= timeB;
  } else {
    aWins = replayA.frames.length >= replayB.frames.length;
  }

  const p1Replay = aWins ? replayA : replayB;
  const p2Replay = aWins ? replayB : replayA;
  const isP1CarA = aWins;

  const p1Time = aWins ? timeA : timeB;
  const p2Time = aWins ? timeB : timeA;

  const p1Name = p1Replay.participantId.toUpperCase();
  const p2Name = p2Replay.participantId.toUpperCase();

  if (winnerName) winnerName.textContent = p1Name;
  if (winnerTimeVal) winnerTimeVal.textContent = formatTime(p1Time);

  if (podP1Name) podP1Name.textContent = p1Name;
  if (podP1Gap) podP1Gap.textContent = 'WINNER';
  if (winnerP1Ind) {
    winnerP1Ind.className = 'pod-indicator ' + (isP1CarA ? 'cyan-bg' : 'gold-bg');
  }

  if (podP2Name) podP2Name.textContent = p2Name;
  if (winnerP2Ind) {
    winnerP2Ind.className = 'pod-indicator ' + (isP1CarA ? 'gold-bg' : 'cyan-bg');
  }

  if (podP2Gap) {
    if (p2Replay.result?.status && p2Replay.result.status.startsWith('dnf')) {
      podP2Gap.textContent = p2Replay.result.status.replace('dnf_', 'DNF ').toUpperCase();
    } else {
      const diff = Math.abs(p2Time - p1Time);
      podP2Gap.textContent = `+${diff.toFixed(3)}s`;
    }
  }

  /* Next track rotation suggestion */
  const nextTrackInfo = {
    'oval': { key: 'flowing', label: 'Flowing' },
    'flowing': { key: 'tight', label: 'Tight Loop' },
    'tight': { key: 'oval', label: 'Oval' }
  };
  const next = nextTrackInfo[currentTrackKey] || nextTrackInfo['oval'];
  if (btnSwitchTrackWinner) {
    btnSwitchTrackWinner.textContent = `🏎️ Next Track: ${next.label}`;
  }

  winnerModal.classList.remove('hidden');
}

/*
 * I switch seamlessly between official circuits: Oval Speedway, Flowing GP Circuit, and Tight Technical Circuit.
 */
async function loadTrack(trackKey) {
  currentTrackKey = trackKey;
  if (trackSwitcher) trackSwitcher.value = trackKey;
  if (winnerModal) winnerModal.classList.add('hidden');
  winnerShownForCurrentRace = false;

  /* Synchronize active track card in hero section */
  document.querySelectorAll('.hero-track-card').forEach(card => {
    card.classList.toggle('active', card.dataset.track === trackKey);
  });

  const trackFiles = {
    oval: ['./racer_a.json', './racer_b.json'],
    flowing: ['./flowing_a.json', './flowing_b.json'],
    tight: ['./tight_a.json', './tight_b.json'],
    monaco: ['./monaco_a.json', './monaco_b.json'],
    spa: ['./spa_a.json', './spa_b.json']
  };

  const files = trackFiles[trackKey] || trackFiles.oval;

  loadStatus.textContent = `Loading Circuit: ${trackKey.toUpperCase()}…`;
  loadingOverlay.style.display = 'flex';
  loadingOverlay.classList.remove('hidden');

  try {
    const [resA, resB] = await Promise.all([
      fetch(files[0]),
      fetch(files[1])
    ]);
    if (!resA.ok || !resB.ok) {
      throw new Error(`Failed to load replay files for circuit "${trackKey}"`);
    }
    const [jsonA, jsonB] = await Promise.all([resA.json(), resB.json()]);

    /* Respect custom uploaded replays if user uploaded them in hero section */
    replayA = customReplayA || parseReplay(jsonA);
    replayB = customReplayB || parseReplay(jsonB);

    /* Clean previous Three.js objects */
    if (trackGroup && scene) {
      scene.remove(trackGroup);
      trackGroup = null;
    }
    if (carModelA && scene) {
      scene.remove(carModelA);
      carModelA = null;
    }
    if (carModelB && scene) {
      scene.remove(carModelB);
      carModelB = null;
    }

    currentTime = 0;
    playing = true;
    btnPlay.textContent = '⏸';
    lastLeader = null;
    lastOffTrackA = false;
    lastOffTrackB = false;

    loadRaceIntoScene();

    loadingOverlay.classList.add('hidden');
    setTimeout(() => { loadingOverlay.style.display = 'none'; }, 400);

    showAlert(`TRACK LOADED: ${trackKey.toUpperCase()} CIRCUIT`, 'info', 3000);
  } catch (err) {
    console.error('Error loading track:', err);
    showAlert(`Error loading track: ${err.message}`, 'danger', 4000);
    loadingOverlay.classList.add('hidden');
    setTimeout(() => { loadingOverlay.style.display = 'none'; }, 400);
  }
}

/* UI Controls and Event Listeners */

/* Play or pause race playback */
btnPlay.addEventListener('click', () => {
  if (liveMode) return;
  soundSystem.ensureResumed();
  if (currentTime >= maxTime) {
    currentTime = 0;
    winnerShownForCurrentRace = false;
    if (winnerModal) winnerModal.classList.add('hidden');
  }
  playing = !playing;
  btnPlay.textContent = playing ? '⏸' : '▶';
});

/* Restart race to beginning */
btnRestart.addEventListener('click', () => {
  if (liveMode) return;
  soundSystem.ensureResumed();
  currentTime = 0;
  playing = false;
  winnerShownForCurrentRace = false;
  if (winnerModal) winnerModal.classList.add('hidden');
  btnPlay.textContent = '▶';
  showAlert('RACE RESET TO START', 'info', 1500);
});

/* Step back or forward 1 second */
if (btnStepBack) {
  btnStepBack.addEventListener('click', () => {
  if (liveMode) return;
    soundSystem.ensureResumed();
    currentTime = Math.max(0, currentTime - 1.0);
    if (currentTime < maxTime - 0.1) {
      winnerShownForCurrentRace = false;
      if (winnerModal) winnerModal.classList.add('hidden');
    }
  });
}
if (btnStepFwd) {
  btnStepFwd.addEventListener('click', () => {
  if (liveMode) return;
    soundSystem.ensureResumed();
    currentTime = Math.min(maxTime, currentTime + 1.0);
  });
}

/* Scrubber timeline seek */
scrubber.addEventListener('input', () => {
  if (liveMode) return;
  soundSystem.ensureResumed();
  currentTime = (scrubber.value / 1000) * maxTime;
  if (currentTime < maxTime - 0.1) {
    winnerShownForCurrentRace = false;
    if (winnerModal) winnerModal.classList.add('hidden');
  }
  playing = false;
  btnPlay.textContent = '▶';
});

/* Playback speed presets */
document.querySelectorAll('.speed-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    soundSystem.ensureResumed();
    playbackSpeed = parseFloat(btn.dataset.speed);
    document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  });
});

/* Camera preset buttons */
document.querySelectorAll('.cam-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    soundSystem.ensureResumed();
    cameraMode = btn.dataset.cam;
    document.querySelectorAll('.cam-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  });
});

/* Audio sound toggle button */
if (btnAudio) {
  btnAudio.addEventListener('click', () => {
    soundSystem.ensureResumed();
    const isUnmuted = soundSystem.toggleMute();
    btnAudio.textContent = isUnmuted ? '🔊 Sound ON' : '🔇 Sound MUTED';
    btnAudio.classList.toggle('active', isUnmuted);
    showAlert(isUnmuted ? 'F1 SOUND ENGINE & AMBIENCE ACTIVATED' : 'AUDIO MUTED', 'info', 1800);
  });
}

/* Lane separation toggle */
btnLanes.addEventListener('click', () => {
  laneSeparation = !laneSeparation;
  btnLanes.textContent = laneSeparation ? 'Lanes: Dual' : 'Lanes: Raw';
  btnLanes.classList.toggle('active', laneSeparation);
  showAlert(laneSeparation ? 'SIDE-BY-SIDE DUAL LANE ACTIVATED' : 'RAW TELEMETRY COORDINATES ACTIVATED', 'info', 2000);
});

/* Track switcher dropdown menu */
if (trackSwitcher) {
  trackSwitcher.addEventListener('change', (e) => {
  if (liveMode) return;
    soundSystem.ensureResumed();
    loadTrack(e.target.value);
  });
}

/* Winner screen action buttons */
if (btnCloseWinner) {
  btnCloseWinner.addEventListener('click', () => {
    winnerModal.classList.add('hidden');
  });
}

if (btnReplayWinner) {
  btnReplayWinner.addEventListener('click', () => {
    soundSystem.ensureResumed();
    winnerModal.classList.add('hidden');
    currentTime = 0;
    winnerShownForCurrentRace = false;
    playing = true;
    btnPlay.textContent = '⏸';
  });
}

if (btnSwitchTrackWinner) {
  btnSwitchTrackWinner.addEventListener('click', () => {
    soundSystem.ensureResumed();
    winnerModal.classList.add('hidden');
    const nextTrackMap = {
      'oval': 'flowing',
      'flowing': 'tight',
      'tight': 'oval'
    };
    const nextKey = nextTrackMap[currentTrackKey] || 'flowing';
    loadTrack(nextKey);
  });
}

/* Bottom bar load button, opens hero section for racer and circuit selection */
if (btnOpenPicker) {
  btnOpenPicker.addEventListener('click', () => {
    soundSystem.ensureResumed();
    if (isHeroMode) soundSystem.startHeroBGM();
    isHeroMode = true;
    document.body.classList.add('hero-active');
    heroShotIndex = 0;
    heroShotTimer = 0;
    if (heroSection) heroSection.classList.remove('hidden');
    showAlert('OPENED RACER & CIRCUIT SELECTOR', 'info', 2000);
  });
}

if (btnClosePicker && filePicker) {
  btnClosePicker.addEventListener('click', () => {
    filePicker.style.display = 'none';
  });
}

/* Hero section circuit switcher cards */
document.querySelectorAll('.hero-track-card').forEach(card => {
  card.addEventListener('click', () => {
    soundSystem.ensureResumed();
    if (isHeroMode) soundSystem.startHeroBGM();
    const trackKey = card.dataset.track;
    loadTrack(trackKey);
  });
});

/* Hero section racer 1 cyan selection and custom upload */
if (heroSelectRacerA) {
  heroSelectRacerA.addEventListener('change', () => {
    soundSystem.ensureResumed();
    if (isHeroMode) soundSystem.startHeroBGM();
    if (heroSelectRacerA.value === 'custom') {
      heroFileA?.click();
    } else {
      customReplayA = null;
      if (heroFileNameA) heroFileNameA.textContent = 'Using built-in telemetry';
      loadTrack(currentTrackKey);
    }
  });
}

if (heroFileA) {
  heroFileA.addEventListener('change', async (e) => {
    soundSystem.ensureResumed();
    const file = e.target.files[0];
    if (file) {
      try {
        const text = await file.text();
        const parsed = parseReplay(JSON.parse(text));
        customReplayA = parsed;
        replayA = parsed;
        if (heroFileNameA) heroFileNameA.textContent = `Uploaded: ${file.name} (${parsed.participantId})`;
        loadRaceIntoScene();
        showAlert(`CAR 1 LOADED: ${parsed.participantId.toUpperCase()}`, 'info', 2500);
      } catch (err) {
        alert('Invalid replay JSON for Car 1: ' + err.message);
        heroSelectRacerA.value = 'builtin';
        if (heroFileNameA) heroFileNameA.textContent = 'Using built-in telemetry';
      }
    }
  });
}

/* Hero section racer 2 papaya selection and custom upload */
if (heroSelectRacerB) {
  heroSelectRacerB.addEventListener('change', () => {
    soundSystem.ensureResumed();
    if (isHeroMode) soundSystem.startHeroBGM();
    if (heroSelectRacerB.value === 'custom') {
      heroFileB?.click();
    } else {
      customReplayB = null;
      if (heroFileNameB) heroFileNameB.textContent = 'Using built-in telemetry';
      loadTrack(currentTrackKey);
    }
  });
}

if (heroFileB) {
  heroFileB.addEventListener('change', async (e) => {
    soundSystem.ensureResumed();
    const file = e.target.files[0];
    if (file) {
      try {
        const text = await file.text();
        const parsed = parseReplay(JSON.parse(text));
        customReplayB = parsed;
        replayB = parsed;
        if (heroFileNameB) heroFileNameB.textContent = `Uploaded: ${file.name} (${parsed.participantId})`;
        loadRaceIntoScene();
        showAlert(`CAR 2 LOADED: ${parsed.participantId.toUpperCase()}`, 'info', 2500);
      } catch (err) {
        alert('Invalid replay JSON for Car 2: ' + err.message);
        heroSelectRacerB.value = 'builtin';
        if (heroFileNameB) heroFileNameB.textContent = 'Using built-in telemetry';
      }
    }
  });
}

/* Launch Grand Prix broadcast button */
if (btnLaunchGP) {
  btnLaunchGP.addEventListener('click', () => {
    soundSystem.ensureResumed();
    soundSystem.stopHeroBGM(1.2); /* Smoothly fade out Hero theme */
    soundSystem.crowd?.ensurePlaying(); /* Start grandstand crowd ambience as cars take the start */

    isHeroMode = false;
    document.body.classList.remove('hero-active');
    if (heroSection) heroSection.classList.add('hidden');

    /* Reset camera field of view and switch to battle chase cam */
    camera.fov = 48;
    camera.updateProjectionMatrix();

    cameraMode = 'battle';
    document.querySelectorAll('.cam-btn').forEach(b => b.classList.remove('active'));
    document.querySelector('[data-cam="battle"]')?.classList.add('active');

    /* Reset to start line and begin official race */
    currentTime = 0;
    playing = true;
    btnPlay.textContent = '⏸';
    winnerShownForCurrentRace = false;
    if (winnerModal) winnerModal.classList.add('hidden');

    showAlert('🟢 LIGHTS OUT AND AWAY WE GO! RACE STARTED', 'info', 3200);
  });
}

/* Open hero section from bottom bar intro button */
if (btnHeroShow) {
  btnHeroShow.addEventListener('click', () => {
    soundSystem.ensureResumed();
    soundSystem.startHeroBGM();
    isHeroMode = true;
    document.body.classList.add('hero-active');
    heroShotIndex = 0;
    heroShotTimer = 0;
    if (heroSection) heroSection.classList.remove('hidden');

    const firstShot = CINEMATIC_SHOTS[0];
    if (heroCamLabel) heroCamLabel.textContent = `CINEMATIC HIGHLIGHTS: ${firstShot.name}`;
    showAlert('CINEMATIC INTRO & HIGHLIGHTS ACTIVE', 'info', 2200);
  });
}

/* Audio context unlock on initial user gesture */
window.addEventListener('click', () => {
  soundSystem.ensureResumed();
  if (isHeroMode) soundSystem.startHeroBGM();
}, { once: true });
window.addEventListener('touchstart', () => {
  soundSystem.ensureResumed();
  if (isHeroMode) soundSystem.startHeroBGM();
}, { once: true });

/* Keyboard shortcuts for race controls */
document.addEventListener('keydown', (e) => {
  soundSystem.ensureResumed();
  if (e.code === 'Space') { e.preventDefault(); btnPlay.click(); }
  if (e.code === 'KeyR') { btnRestart.click(); }
  if (e.code === 'KeyL') { btnLanes.click(); }
  if (e.code === 'KeyM') { btnAudio?.click(); }
  if (e.code === 'BracketLeft') { btnStepBack?.click(); }
  if (e.code === 'BracketRight') { btnStepFwd?.click(); }
  if (e.code === 'Digit1') document.querySelector('[data-cam="battle"]')?.click();
  if (e.code === 'Digit2') document.querySelector('[data-cam="carA"]')?.click();
  if (e.code === 'Digit3') document.querySelector('[data-cam="carB"]')?.click();
  if (e.code === 'Digit4') document.querySelector('[data-cam="cockpit"]')?.click();
  if (e.code === 'Digit5') document.querySelector('[data-cam="top"]')?.click();
  if (e.code === 'Digit6') document.querySelector('[data-cam="side"]')?.click();
  if (e.code === 'ArrowRight') { currentTime = Math.min(currentTime + 1 / 30, maxTime); }
  if (e.code === 'ArrowLeft') { currentTime = Math.max(currentTime - 1 / 30, 0); }
  if (e.code === 'ArrowUp') {
    const speeds = [0.5, 1, 2, 4];
    const idx = speeds.indexOf(playbackSpeed);
    if (idx < speeds.length - 1) {
      document.querySelector(`[data-speed="${speeds[idx + 1]}"]`)?.click();
    }
  }
  if (e.code === 'ArrowDown') {
    const speeds = [0.5, 1, 2, 4];
    const idx = speeds.indexOf(playbackSpeed);
    if (idx > 0) {
      document.querySelector(`[data-speed="${speeds[idx - 1]}"]`)?.click();
    }
  }
});

/* File picker inputs */
if (fileA && fileB && btnLoad) {
  function checkFilesReady() {
    btnLoad.disabled = !(fileA.files.length && fileB.files.length);
  }
  fileA.addEventListener('change', checkFilesReady);
  fileB.addEventListener('change', checkFilesReady);
}

if (btnLoad && fileA && fileB) {
  btnLoad.addEventListener('click', async () => {
    if (pickerError) pickerError.textContent = '';
    try {
      const textA = await fileA.files[0].text();
      const textB = await fileB.files[0].text();
      replayA = parseReplay(JSON.parse(textA));
      replayB = parseReplay(JSON.parse(textB));

      if (replayA.track.meta.content_hash !== replayB.track.meta.content_hash) {
        throw new Error('Both replays must use the exact same track geometry.');
      }

      if (filePicker) filePicker.style.display = 'none';
      startRace();
    } catch (err) {
      if (pickerError) pickerError.textContent = err.message;
    }
  });
}

/* Quick launch sample replays if present */
if (btnQuickLoad) {
  btnQuickLoad.addEventListener('click', async () => {
    if (pickerError) pickerError.textContent = '';
    try {
      if (loadStatus) loadStatus.textContent = 'Fetching built-in replays…';
      const [resA, resB] = await Promise.all([
        fetch('./racer_a.json'),
        fetch('./racer_b.json')
      ]);
      if (!resA.ok || !resB.ok) {
        throw new Error('Failed to load sample replays from server.');
      }
      const [jsonA, jsonB] = await Promise.all([resA.json(), resB.json()]);
      replayA = parseReplay(jsonA);
      replayB = parseReplay(jsonB);

      if (filePicker) filePicker.style.display = 'none';
      startRace();
    } catch (err) {
      if (pickerError) pickerError.textContent = 'Could not load built-in files: ' + err.message;
    }
  });
}

/* Start the race and transition loading overlay */
function startRace() {
  if (loadStatus) loadStatus.textContent = 'Building 3D Circuit & F1 Models…';

  function dismissLoader() {
    if (loadingOverlay) {
      loadingOverlay.classList.add('hidden');
      setTimeout(() => { loadingOverlay.style.display = 'none'; }, 300);
    }
  }

  requestAnimationFrame(() => {
    try {
      if (!scene) initThreeJS();
      loadRaceIntoScene();
    } catch (err) {
      console.error('Scene creation error:', err);
    } finally {
      dismissLoader();
    }

    if (!isAnimating) {
      isAnimating = true;
      animate();
    }
  });

  /* Guaranteed safety fallback, hide loader after 1200ms regardless of any asset delays */
  setTimeout(dismissLoader, 1200);
}


if (btnLiveConnect) {
  btnLiveConnect.addEventListener('click', () => {
    soundSystem.ensureResumed();
    const url = liveUrl?.value?.trim();
    if (!url) {
      showAlert('PLEASE ENTER A VALID LIVE WEBSOCKET URL', 'warn', 2500);
      return;
    }
    if (pickerError) pickerError.textContent = '';
    if (filePicker) filePicker.style.display = 'none';
    startLiveConnection(url);
  });
}

/* Automatic launch on page load */
async function autoInit() {
  const liveQuery = new URLSearchParams(window.location.search).get('live');
  if (liveQuery) {
    startLiveConnection(liveQuery);
    return;
  }

  function dismissLoader() {
    if (loadingOverlay) {
      loadingOverlay.classList.add('hidden');
      setTimeout(() => { loadingOverlay.style.display = 'none'; }, 300);
    }
    if (filePicker) filePicker.style.display = 'flex';
    if (heroSection) heroSection.classList.remove('hidden');
  }

  /* Safety timer guaranteeing the loading overlay will never stay stuck on screen */
  const safetyTimer = setTimeout(dismissLoader, 1500);

  /* 1. Check embedded data for standalone bundled HTML distributions */
  const dataA = document.getElementById('replay-data-a');
  const dataB = document.getElementById('replay-data-b');
  if (dataA && dataB) {
    try {
      replayA = parseReplay(JSON.parse(dataA.textContent));
      replayB = parseReplay(JSON.parse(dataB.textContent));
      clearTimeout(safetyTimer);
      startRace();
      return;
    } catch (e) {
      console.warn('Embedded data parse failed:', e);
    }
  }

  /* 2. Attempt fetching default sample files */
  try {
    const fetchWithTimeout = (url, ms = 2000) => {
      const controller = new AbortController();
      const id = setTimeout(() => controller.abort(), ms);
      return fetch(url, { signal: controller.signal }).finally(() => clearTimeout(id));
    };

    const [resA, resB] = await Promise.all([
      fetchWithTimeout('./racer_a.json'),
      fetchWithTimeout('./racer_b.json')
    ]);
    if (resA.ok && resB.ok) {
      const [jsonA, jsonB] = await Promise.all([resA.json(), resB.json()]);
      replayA = parseReplay(jsonA);
      replayB = parseReplay(jsonB);
      clearTimeout(safetyTimer);
      startRace();
      return;
    }
  } catch (e) {
    console.info('Auto fetch bypassed, presenting hero section:', e);
  }

  /* 3. Fallback, dismiss loading overlay and present the hero section */
  clearTimeout(safetyTimer);
  dismissLoader();
}

autoInit();
