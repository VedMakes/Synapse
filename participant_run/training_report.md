# Participant training run

This is the provisional local participant workflow. The budget and event settings are not frozen public rules.

## Current state

- Release: `synapse-local-participant-prototype`
- Tracks: `synthetic_flowing, synthetic_oval, synthetic_tight_loop`
- Budget used: `2560` / `10000` transitions
- Budget remaining: `7440`
- Updates: `10`
- Model: fixed two-layer MLP `[64, 64]`
- Resume note: `optimizer state is reset on process restart in this prototype`

## Update metrics

| Update | Steps used | Mean return | Lap completion | Mean speed | Crash rate | Off-track rate | Approx. KL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 256 | -10.371265 | 0.000000 | 14.181422 | 1.000000 | 0.144531 | 0.000156 |
| 2 | 512 | -10.339408 | 0.000000 | 13.992989 | 1.000000 | 0.156250 | 0.000159 |
| 3 | 768 | -10.405520 | 0.000000 | 13.938735 | 1.000000 | 0.160156 | 0.000124 |
| 4 | 1024 | -10.564601 | 0.000000 | 10.248778 | 1.000000 | 0.269531 | 0.000033 |
| 5 | 1280 | -11.635901 | 0.000000 | 11.376214 | 1.000000 | 0.421875 | 0.000023 |
| 6 | 1536 | -10.442667 | 0.000000 | 10.587030 | 1.000000 | 0.113281 | 0.000018 |
| 7 | 1792 | -10.809564 | 0.000000 | 10.218934 | 1.000000 | 0.355469 | 0.000032 |
| 8 | 2048 | -10.850035 | 0.000000 | 12.819483 | 1.000000 | 0.417969 | 0.000068 |
| 9 | 2304 | -11.104519 | 0.000000 | 7.281174 | 1.000000 | 0.078125 | 0.000021 |
| 10 | 2560 | -11.841017 | 0.000000 | 8.816908 | 1.000000 | 0.429688 | 0.000060 |
