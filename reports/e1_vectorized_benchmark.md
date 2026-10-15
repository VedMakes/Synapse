# E1 Vectorized Environment Benchmark

- Date (UTC): 2026-10-15T19:06:14.272878+00:00
- Host: Linux-7.0.0-38-generic-x86_64-with-glibc2.36
- Python: 3.11.2
- NumPy: 2.4.6
- Track: `synthetic_oval` (273a66d19cfe9b01b6c3de29d91f432b051878d4f279acd63dddf035c4dfa374)
- Tick rate: 30 Hz
- Measured transitions per row: 10
- Sensor/projection column is full environment time minus batched physics-only time; it includes geometry, reward, and observation work.

| n_envs | environment steps/s | policy + environment steps/s | physics-only steps/s | geometry/reward/observation overhead ms/env-step |
|---:|---:|---:|---:|---:|
| 1 | 68.99 | 65.58 | 10074.00 | 14.395 |
| 2 | 70.93 | 66.83 | 22334.76 | 14.053 |
| 4 | 71.73 | 71.03 | 30872.06 | 13.909 |
| 8 | 72.02 | 71.39 | 84689.60 | 13.873 |
| 16 | 68.75 | 69.01 | 154684.87 | 14.539 |

## Interpretation

These are local measurements for the current Python process and fixture track, not a portable performance guarantee. The full environment currently retains per-car geometry calls for scalar parity. Optimize raycasts/projection only after comparing this baseline with the required training workload.
