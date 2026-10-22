# SYNAPSE 2.5D Dual-Car Race Renderer

A modern, broadcast-grade **2.5D WebGL F1 TV Chase Viewer** built with Three.js for the SYNAPSE Autonomous Racing platform.

---

## 🌟 Visual & Environmental Features

- **Authentic Sunlit F1 Grid:** Bright, daytime Formula 1 atmosphere with warm sunlight (`3.2` intensity), vivid azure sky, soft horizon haze, and zero dark black voids.
- **Clear Track Geometry (Zero Obstructions):**
  - High-clearance start/finish gantry ($9.5\text{m}$ clearance) with pillars placed strictly **3.5m behind the outer safety barriers**.
  - All perimeter stadium floodlights and spectator grandstands are positioned **18m outside the outer barriers** on the open countryside turf — completely off the racing tarmac and runoffs.
- **Realistic Circuit Materials:**
  - **Asphalt Road Surface:** Authentic asphalt grey (`#363b46`) with subtle specular highlights, sharp white boundary lines, dashed centerline, and painted F1 grid slot boxes (Slot 1 & Slot 2).
  - **Runoff Traps:** Warm sandy golden gravel traps (`#d9b372`) and F1 Royal Blue painted runoff aprons (`#1e3a8a`).
  - **Apex Curbs:** Vibrant red (`#e10600`) and pure white (`#ffffff`) 3D apex curbing.
  - **Safety Barriers:** Clean white/silver TecPro barrier walls (`#f8fafc`) with vibrant blue safety railings.
  - **Infield & Countryside:** Lush summer green grass turf (`#3d8b37`).
- **High-Detail F1 Car Models:**
  - Tapered aerodynamic nose cone with carbon fiber nose tip.
  - Sculpted sidepods with dark radiator inlet grilles and undercut floors.
  - 3D curved titanium Halo safety ring arching over the cockpit.
  - Driver helmet with glossy reflective gold/dark visor.
  - Multi-tier front wing with endplate canards, high-downforce rear wing with DRS flap, and pulsing red rain safety LED.
  - Wide Pirelli slick tires with yellow sidewall stripes and team-colored wheel rims.
  - **Car A (Racer Alpha):** Pearlescent Electric Cyan (`#00f0ff`) with gloss clearcoat and white aerodynamic flaps.
  - **Car B (Racer Beta):** Metallic Solar Papaya Orange (`#ff5500`) with gold aerodynamic flaps.
- **Dynamic X-Ray Overlap Transparency & Glowing Wireframe Matrix:**
  - If two cars come within $2.8\text{m}$ of each other (or in Raw Telemetry mode where they occupy the exact same coordinate):
    - Both car bodies smoothly become **semi-transparent** (~45% opacity).
    - Glowing **holographic wireframe matrices** illuminate around both cars (Cyan for Car A, Solar Gold for Car B), allowing viewers to clearly see both cars simultaneously without clipping!
    - The timing tower displays an animated `⚡ X-RAY OVERLAP MODE` badge.
    - When cars separate ($> 2.8\text{m}$), they smoothly transition back to 100% solid paint with no wireframe.
- **Dynamic Chase Camera System:**
  - `[1] Battle Cam`: Smoothly tracks the centroid between both cars, zooming in during close wheel-to-wheel battles and pulling back when they separate.
  - `[2] Car A Chase`: Locked follow camera behind Racer Alpha.
  - `[3] Car B Chase`: Locked follow camera behind Racer Beta.
  - `[4] Cockpit Cam`: First-person nose/halo camera for high-speed perspective.
  - `[5] Top Drone Cam`: Overhead tactical overview of the full circuit.
- **Clean 2-Row Bottom Broadcast Control Bar:**
  - Row 1: Full-width scrubbable progress timeline with high-precision timecode indicators (`00:00.000` elapsed and total duration).
  - Row 2: Four distinct control groups:
    - *Transport:* Restart `⏮`, Step Back `-1s`, Play/Pause `▶`/`⏸`, Step Forward `+1s`.
    - *Speed:* Preset speed pills `0.5×`, `1×`, `2×`, `4×`.
    - *Cameras:* `Battle [1]`, `Car A [2]`, `Car B [3]`, `Cockpit [4]`, `Top [5]`.
    - *Auxiliary:* `Lanes: Dual (L)`, `🏁 Track Select` dropdown, and `📂 Load` custom replay button.
- **Winner Victory Screen & Podium Modal:**
  - When the race reaches the checkered flag, an official F1 victory podium modal pops up displaying:
    - 🏆 Official P1 Winner banner and winning participant name
    - Official race completion time (e.g. `00:28.767`)
    - Detailed podium leaderboard table with P1 and P2 ranks, team color indicators, and finish gap / DNF status
    - Interactive action buttons: `⏮ Replay Race` and `🏎️ Next Track` to cycle through circuits!
- **Instant Circuit Switching:**
  - Switch between circuits (`🏁 Oval`, `🏁 Flowing`, `🏁 Tight Loop`) instantly using the track selector in the bottom control bar or the victory screen without restarting the server.
- **Top-Right Radar Minimap:** Live circuit radar with track contour, glowing car blips with directional heading needles, and dynamic camera field-of-view cone.
- **F1 Live Timing Tower & Telemetry:**
  - Real-time P1/P2 leaderboard with interval gaps and leader tags.
  - Dynamic sector indicator (`SEC 1`, `SEC 2`, `SEC 3`) and lap tracker (`LAP 1/1`).
  - Real-time digital speedometer (km/h), throttle & brake bars, steering wheel angle in degrees (e.g. `+14.2°`), F1 gear indicator (`G1`–`G8`), and lateral G-force meter (e.g. `1.8 G`).
  - Floating broadcast alerts for track limits (`OFF TRACK`), collisions (`BARRIER CONTACT`), lead changes (`OVERTAKE`), and checkered flag.

---

## 🚀 Quick Start Guide: How to Run the Renderer

All commands should be executed from the `d:\synapse_shit\Synapse` root directory in PowerShell or Bash:

### Step 1: Start the Local Web Server

```powershell
python -m http.server 8765 --bind 127.0.0.1 --directory renderer
```

### Step 2: Open in Your Browser

Open Chrome, Edge, or Firefox and navigate to:

👉 **[http://127.0.0.1:8765](http://127.0.0.1:8765)**

*(The built-in sample replays load automatically on launch!)*

### Live Python race mode

The renderer also has a read-only live adapter. Start the Python live runner
from the repository root, then open the renderer with the printed WebSocket
URL:

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory renderer
.venv/bin/python tools/run_live_onnx.py \
  --entrant racer-a submissions/racer-a \
  --entrant racer-b submissions/racer-b \
  --track tracks/train/synthetic_oval.json \
  --config config/simulation.json \
  --output-dir artifacts/live/oval
```

Open `http://127.0.0.1:8765/?live=ws://127.0.0.1:8766/live/v1`, or paste that
WebSocket URL into **Connect Live**. Python owns every action, physics step,
event, and result. The browser only receives state packets, interpolates
received poses for presentation, and shows `STALE` if packets stop arriving.
Replay controls are disabled in live mode. The live stream is not a replay and
does not send actions back to Python.

---

## 🏎️ How to Generate Dual-Car Replays for Any Track

You can generate fresh head-to-head evaluation replays for any circuit using `tools/export_replay.py`:

### A. Synthetic Oval Track
```powershell
python tools/export_replay.py --scripted --track tracks/train/synthetic_oval.json --config config/simulation.json --participant-id racer-alpha --output artifacts/replay/racer_a.json
python tools/export_replay.py --scripted --track tracks/train/synthetic_oval.json --config config/simulation.json --participant-id racer-beta --steps 650 --output artifacts/replay/racer_b.json
```

### B. Synthetic Flowing Track
```powershell
python tools/export_replay.py --scripted --track tracks/train/synthetic_flowing.json --config config/simulation.json --participant-id flowing-alpha --output artifacts/replay/flowing_a.json
python tools/export_replay.py --scripted --track tracks/train/synthetic_flowing.json --config config/simulation.json --participant-id flowing-beta --steps 500 --output artifacts/replay/flowing_b.json
```

### C. Synthetic Tight Loop Track
```powershell
python tools/export_replay.py --scripted --track tracks/train/synthetic_tight_loop.json --config config/simulation.json --participant-id tight-alpha --output artifacts/replay/tight_a.json
python tools/export_replay.py --scripted --track tracks/train/synthetic_tight_loop.json --config config/simulation.json --participant-id tight-beta --steps 450 --output artifacts/replay/tight_b.json
```

---

## 📦 How to Bundle Standalone Offline HTML Files (Phase 7)

To generate self-contained, single-file HTML replays that open anywhere on any device without running a server:

```powershell
# Bundle Oval Track Race
python renderer/build_replay.py --racer-a artifacts/replay/racer_a.json --racer-b artifacts/replay/racer_b.json --output artifacts/replay/battle_oval.html

# Bundle Flowing Track Race
python renderer/build_replay.py --racer-a artifacts/replay/flowing_a.json --racer-b artifacts/replay/flowing_b.json --output artifacts/replay/battle_flowing.html

# Bundle Tight Loop Race
python renderer/build_replay.py --racer-a artifacts/replay/tight_a.json --racer-b artifacts/replay/tight_b.json --output artifacts/replay/battle_tight.html
```

You can double-click any generated `.html` file in `artifacts/replay/` to run it offline!

---

## 🎮 Keyboard Shortcuts & Controls Reference

| Control | Shortcut | Description |
|---|---|---|
| **Play / Pause** | `Space` | Toggle playback |
| **Reset to Start** | `Key R` | Rewind simulation to Tick 0 |
| **Lane Separation** | `Key L` | Toggle Side-by-Side Dual Lane vs Raw Telemetry (test overlap wireframe!) |
| **Battle Cam** | `1` | Default broadcast chase camera tracking both cars |
| **Car A Chase** | `2` | Tight follow camera behind Racer Alpha |
| **Car B Chase** | `3` | Tight follow camera behind Racer Beta |
| **Nose / Cockpit** | `4` | First-person nose/halo perspective |
| **Top Drone View** | `5` | High-altitude tactical overview of the circuit |
| **Step 1s Back** | `[` | Step backwards by 1.0 second |
| **Step 1s Forward**| `]` | Step forward by 1.0 second |
| **Step 1 Frame** | `←` / `→` | Step backwards / forwards by 1 frame (1/30s) |
| **Playback Speed** | `↑` / `↓` | Increase / decrease playback speed (0.5×, 1×, 2×, 4×) |
| **Load Replays** | `📂 Load` | Open modal to select custom replay JSON files |
