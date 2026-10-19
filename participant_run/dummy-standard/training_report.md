# Reference baseline training run

This is an optional reference baseline. It is not the participant model or event contract.

## Current state

- Release: `synapse-local-participant-prototype`
- Tracks: `synthetic_flowing, synthetic_oval, synthetic_tight_loop`
- Budget used: `16` / `16` transitions
- Budget remaining: `0`
- Updates: `1`
- Reference model: fixed two-layer MLP `[64, 64]`
- Resume note: `optimizer state is reset on process restart in this prototype`

## Update metrics

| Update | Steps used | Mean return | Lap completion | Mean speed | Crash rate | Off-track rate | Approx. KL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | n/a | n/a | 8.460408 | n/a | 0.000000 | 0.004528 |
