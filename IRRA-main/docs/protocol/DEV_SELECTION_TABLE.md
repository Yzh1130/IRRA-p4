# Development-set hyperparameter selection

All results below use only the locked `val` split. R1 values are percentage points.

## MRAF target temperature

Each seed value is its mean R1 over the ten degraded scenarios (clean excluded).
The degraded and clean summaries are the mean ± sample standard deviation across three seeds.

| T_r | role | seed 1 | seed 2 | seed 3 | degraded R1 mean ± SD | clean R1 mean ± SD | selected for E |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0.1 | reference | 61.6015 | 62.6773 | 60.6601 | 61.6463 ± 1.0093 | 69.2339 ± 0.7369 | no |
| 0.2 | candidate | 63.1296 | 62.0905 | 62.8362 | 62.6854 ± 0.5357 | 69.1524 ± 0.8321 | yes |
| 0.3 | candidate | 63.0807 | 62.7017 | 61.3692 | 62.3839 ± 0.8989 | 68.5412 ± 1.1359 | no |
| 0.5 | candidate | 63.3374 | 62.0782 | 61.7237 | 62.3798 ± 0.8481 | 67.9707 ± 1.5608 | no |

Selection compares candidate means after `ROUND_HALF_UP` rounding to four decimal places. If the top-minus-second difference is strictly below 0.3000 percentage points, the smaller of those two temperatures is used.

Observed top-minus-second difference: 0.3015; tie rule applied: no.

## Utility prior

`ir_missing` is RGB-only and estimates RGB utility; `rgb_missing` is thermal-only and estimates IR utility.

| modality | source scenario | seed 1 | seed 2 | seed 3 | raw mean | normalized |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RGB | `ir_missing` | 64.7922 | 64.3032 | 63.0807 | 64.0587 | 0.72845227 |
| IR | `rgb_missing` | 21.8826 | 25.3056 | 24.4499 | 23.8794 | 0.27154773 |

## Audit provenance

- Development split MD5: `80356d48d3c78cfa2b489c34c322d373`
- Development metrics MD5: `58bdb7e33a924954db33ffa27ac609e8`
- Seeds: `1, 2, 3`
- Frozen at: `2026-09-06T13:16:12Z`
