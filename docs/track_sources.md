# Public reference track sources

The participant-facing development maps `monaco.json` and `spa.json` are
SYNAPSE geometry fixtures derived from public centerline references. They are
not official circuit assets and do not include logos, textures, models, or
other circuit-owned material.

## Source and transformation

- Source repository: [bacinger/f1-circuits](https://github.com/bacinger/f1-circuits)
- Source license: MIT, verified from the repository metadata and `LICENSE.md`.
- Monaco reference: current Grand Prix centerline, target lap length 3,337 m.
- Spa-Francorchamps reference: current Grand Prix centerline, target lap length
  7,004 m.

The checked-in JSON files contain a compact centerline extraction, projected to
local metres, uniformly resampled to 512 points, and uniformly scaled to the
target lap length. SYNAPSE generates the 10 m road, 5 m runoff, and barrier
boundaries from that centerline. The content hash covers the resulting JSON.

The published dimensions and turn counts were cross-checked against the
current public circuit summaries for [Monaco](https://en.wikipedia.org/wiki/Circuit_de_Monaco)
and [Spa-Francorchamps](https://en.wikipedia.org/wiki/Circuit_de_Spa-Francorchamps).
