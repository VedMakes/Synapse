# Reference baseline training run

This is an optional reference baseline. It is not the participant model or event contract.

## Current state

- Release: `synapse-local-participant-prototype`
- Tracks: `synthetic_flowing, synthetic_oval, synthetic_tight_loop`
- Budget used: `4` / `4` transitions
- Budget remaining: `0`
- Updates: `2`
- Reference model: fixed two-layer MLP `[64, 64]`
- Resume note: `optimizer state is reset on process restart in this prototype`

## Update metrics

| Update | Steps used | Mean return | Lap completion | Mean speed | Crash rate | Off-track rate | Approx. KL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | n/a | n/a | 0.038401 | n/a | 0.000000 | 0.004005 |
| 2 | 4 | n/a | n/a | 0.124492 | n/a | 0.000000 | 0.001288 |
