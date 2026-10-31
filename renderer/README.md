# SYNAPSE 2.5D Dual-Car Race Renderer

A modern, broadcast-grade **2.5D WebGL F1 TV Chase Viewer** built with Three.js for the SYNAPSE Autonomous Racing platform.

---

## 🔒 Competition Privacy and Replay Security

> **IMPORTANT FOR ORGANIZERS:**
> Model telemetry files (`*.json`) contain full 30 Hz observation, velocity, and action pairs (`[steering_command, throttle_brake]`).
> Releasing these files allows participants to clone the driving policy via behavioral cloning (imitation learning) and bypass reinforcement learning training.
> 
> All `.json` replays and telemetry dumps are strictly gitignored under `renderer/.gitignore`.
> Organizers can safely keep private participant replays on their local machine inside `renderer/keepin/` or `renderer/replays/`, which will never be tracked or committed to GitHub.

---

## 🌟 Visual & Environmental Features

- **F1 Grand Prix Hero Section & Cinematic Highlight Showcase:**
  - **Broadcast Hero Overlay:** Pre-race launchpad displayed on initial load.
  - **Circuit Selection Grid:** Direct visual card selector for official circuits (**Monaco**, **Spa-Francorchamps**, **Flowing**, **Oval**).
  - **Hero BGM Audio Engine:** Background music player with procedural synthesizer fallback.
  - **6 Aesthetic Cinematic Racing Highlights & Cuts:** Ground cam, telephoto straight zoom, lateral pan, drone sweep, diffuser chase, grandstand POV.
- **Authentic Circuit Atmosphere & Enhanced Realism:**
  - Bright daytime or illuminated night grand prix lighting with zero dark black voids.
  - **Circuit de Monaco:** Night lighting, harbor yachts, safety boats, and urban skyline ambient glow.
  - **Spa-Francorchamps:** Ardennes pine forest canopy, rolling hill topography, and speed telemetry cones.
  - **Procedural Asphalt Bump Map:** Road grain texture with specular sheen.
  - **Glowing Carbon-Ceramic Brake Discs:** Rotors glow hot orange-red under hard braking.
  - **Beveled Apex Rumble Curbs:** 3D extruded curbs with alternating FIA red and white styling.
  - **Distance Braking Boards:** FIA standard 150m, 100m, and 50m marker boards placed on outer barrier walls.
  - **Low-Poly 3D Audience & Grandstands:** Color-coded spectators in team apparel and fluttering team flags.
  - **Strict Safety Clearances:** All roadside structures, gantries, and grandstands are positioned safely outside barrier runoffs without tarmac clipping.
- **High-Detail F1 Car Models:**
  - Procedural liveries, team branding roundels, carbon wing mirrors, multi-tier front wings, underbody diffusers, titanium exhausts, Halo safety rings, and glowing rain LEDs.
  - **Car A (Cyan):** Pearlescent Electric Cyan livery (`#00f0ff`).
  - **Car B (Papaya):** Metallic Solar Papaya Orange livery (`#ff5500`).
- **Real-Time Sound Engine & Custom Audio Asset Support:**
  - Built-in multi-oscillator V6 turbo-hybrid engine roar, crowd cheering swells, and broadcast ambient beds.
  - Optional custom WAV/MP3 drop-in assets supported in `renderer/assets/` (`hero_bgm.mp3`, `crowd_noise.mp3`, `bg_sound.mp3`, `engine_sound.mp3`).
- **Neo-Brutalist F1 TV Broadcast Graphics UI:**
  - F1 steering wheel 15-LED sequential shift light HUD with redline alert.
  - DRS activation indicator pill on high-speed straights.
  - Official F1 Timing Tower with live driver codes (`ALF`, `BET`), interval gaps, and mini-sector status dots.
  - Provisional classification victory modal with podium breakdown, lap records, and instant circuit rotation.
  - Live minimap radar with car directional needles and camera viewing frustum cone.
- **Dynamic Chase Camera System:**
  - `[1] Battle Cam`: Smoothly tracks the centroid between both cars.
  - `[2] Car A Chase`: Follow camera behind Car A.
  - `[3] Car B Chase`: Follow camera behind Car B.
  - `[4] Cockpit Cam`: First-person nose/halo camera.
  - `[5] Top Drone Cam`: High-altitude tactical overhead view.
  - `[6] Side Trackside Cam`: Low-slung dynamic TV broadcast side-profile camera.

---

## 🚀 How to Run the Races Live

Run all commands from the **repository root directory**.

### Step 1: Start the Local Presentation Web Server

The 3D renderer runs locally on port `8765`.

#### Windows (PowerShell or Command Prompt):
```powershell
python -m http.server 8765 --bind 127.0.0.1 --directory renderer
```
*Or double-click `run_renderer.bat` in the repository root.*

#### Linux (Bash):
```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory renderer
```

Open your browser to: **[http://127.0.0.1:8765](http://127.0.0.1:8765)**

---

### Mode A: Tournament Pre-Replay Workflow (Recommended for Live Events)

This is the safest and cleanest approach for live tournaments. You simulate each model once beforehand (full lap without step cutoff), and during the live knockout matches you pair any two teams with zero loading delay.

#### 1. Simulate and Generate Replays for All Participant Packages

#### Windows:
```powershell
python renderer/tournament.py simulate --all artifacts/onnx-packages --track monaco
```
*Or double-click `renderer/run_tournament.bat` and select Option 1.*

#### Linux:
```bash
python3 renderer/tournament.py simulate --all artifacts/onnx-packages --track monaco
```

This runs full authoritative physics until each model finishes the lap or has an incident, saving replays to `renderer/replays/<team>_monaco.json` and printing the official leaderboard.

#### 2. Stage Any Knockout Match Instantly (0.1s Loading)

#### Windows:
```powershell
python renderer/tournament.py match team_alpha team_beta --track monaco --open-browser
```
*Or double-click `renderer/run_tournament.bat` and select Option 2.*

#### Linux:
```bash
python3 renderer/tournament.py match team_alpha team_beta --track monaco --open-browser
```

This immediately stages `match_a.json` and `match_b.json`, calculates the provisional classification, and opens the 3D duel in your browser at:
`http://127.0.0.1:8765/?match=true`

#### 3. View Current Tournament Leaderboard

#### Windows:
```powershell
python renderer/tournament.py leaderboard --track monaco
```

#### Linux:
```bash
python3 renderer/tournament.py leaderboard --track monaco
```

---

### Mode B: Real-Time WebSocket Live Telecast

In this mode, Python runs physics in real time and streams state packets over WebSockets to the 3D viewer.

#### 1. Start the Live Telecast Engine

#### Windows:
```powershell
python renderer/run_live_broadcast.py --alpha artifacts/onnx-packages/team_alpha/submission --beta artifacts/onnx-packages/team_beta/submission --track monaco --open-browser
```
*Or double-click `renderer/run_live_broadcast.bat`.*

#### Linux:
```bash
python3 renderer/run_live_broadcast.py \
  --alpha artifacts/onnx-packages/team_alpha/submission \
  --beta artifacts/onnx-packages/team_beta/submission \
  --track monaco \
  --open-browser
```

#### 2. Browser Presentation Endpoint
Navigate to:
`http://127.0.0.1:8765/?live=ws://127.0.0.1:8767/live/v1`

Python owns physics, steering commands, and timing. The browser is purely a presentation viewer receiving 30 Hz WebSocket frames.

---

### Mode C: Direct Knockout Match Simulation

If you want to simulate a single head-to-head battle directly from two submission folders:

#### Windows:
```powershell
python renderer/run_match.py --alpha artifacts/onnx-packages/team_alpha/submission --beta artifacts/onnx-packages/team_beta/submission --track monaco --open-browser
```
*Or double-click `renderer/run_match.bat`.*

#### Linux:
```bash
python3 renderer/run_match.py \
  --alpha artifacts/onnx-packages/team_alpha/submission \
  --beta artifacts/onnx-packages/team_beta/submission \
  --track monaco \
  --open-browser
```

---

## 🎮 Keyboard Shortcuts & Broadcast Controls

| Control | Shortcut | Description |
|---|---|---|
| **Play / Pause** | `Space` | Toggle playback |
| **Sound Mute / Unmute** | `Key M` | Toggle F1 engine, crowd, and TV broadcast audio |
| **Reset to Start** | `Key R` | Rewind simulation to Tick 0 |
| **Lane Separation** | `Key L` | Toggle Side-by-Side Dual Lane vs Raw Telemetry |
| **Battle Cam** | `1` | Broadcast chase camera tracking the centroid between both cars |
| **Car A Chase** | `2` | Follow camera behind Car A |
| **Car B Chase** | `3` | Follow camera behind Car B |
| **Nose / Cockpit** | `4` | First-person nose/halo perspective |
| **Top Drone View** | `5` | High-altitude tactical overview of the circuit |
| **Side Trackside View**| `6` | Low-slung dynamic TV broadcast side-profile tracking cam |
| **Step 1s Back** | `[` | Step backwards by 1.0 second |
| **Step 1s Forward**| `]` | Step forward by 1.0 second |
| **Step 1 Frame** | `Left` / `Right` | Step backwards / forwards by 1 frame (1/30s) |
| **Playback Speed** | `Up` / `Down` | Adjust speed (0.5x, 1x, 2x, 4x) |
| **Intro Showcase** | `Intro` Button | Re-open the cinematic hero showcase overlay |
| **Custom Replay Load** | `Load` Button | Upload any custom JSON replay file directly into Car 1 or Car 2 |
