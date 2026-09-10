# RGBNT201-PEDES external-validation precondition

This report uses only configuration A on the locked `dev2` (`val`) split.
R1 values are percentage points; relative drops are percentages of each seed's clean R1.

## Modality utility

`ir_missing` is RGB-only and estimates `û_rgb`; `rgb_missing` is IR-only and estimates `û_ir`.

| estimate | source scenario | seed 1 R1 | seed 2 R1 | seed 3 R1 | mean R1 |
| --- | --- | ---: | ---: | ---: | ---: |
| `û_rgb` | `ir_missing` | 51.3513 | 49.2176 | 52.7738 | 51.1143 |
| `û_ir` | `rgb_missing` | 20.1991 | 24.3243 | 20.7681 | 21.7639 |

**Measured utility ratio `û_rgb / û_ir`: 2.3486.**

## Noise sensitivity relative to clean

For every seed and scenario, relative R1 drop is `100 × (R1_clean - R1_noisy) / R1_clean`; a negative value means the noisy score was higher than clean.

| scenario | perturbed modality | seed 1 drop (%) | seed 2 drop (%) | seed 3 drop (%) | mean drop (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| `rgb_noise_005` | RGB | -0.2718 | -1.6949 | -1.9178 | -1.2948 |
| `ir_noise_005` | IR | 0.8152 | -0.5650 | -1.0959 | -0.2819 |
| `rgb_noise_010` | RGB | -0.2718 | -5.9322 | -0.8219 | -2.3420 |
| `ir_noise_010` | IR | 2.1739 | -1.6949 | -1.6438 | -0.3883 |

Matched-strength comparison:

| noise strength | RGB mean relative drop (%) | IR mean relative drop (%) | more sensitive modality |
| ---: | ---: | ---: | --- |
| 0.05 | -1.2948 | -0.2819 | IR |
| 0.10 | -2.3420 | -0.3883 | IR |

- Utility ordering: **RGB > IR**.
- Mean noise-sensitivity ordering across 0.05 and 0.10: **IR > RGB** (RGB -1.8184%, IR -0.3351%).
- Sensitivity and utility orderings are opposite: **yes**.

## Cross-dataset utility-ratio anchors

| dataset | split | `û_rgb / û_ir` | source |
| --- | --- | ---: | --- |
| RGBNT201-PEDES | dev2 (`val`) | 2.3486 | measured above |
| RGBT-PEDES | test | 3.80 | locked comparison anchor |
| RGBT-PEDES | development | 2.68 | locked comparison anchor |

## Audit provenance

- Metrics source: `/root/autodl-tmp/EXT_GRID/metrics.tsv`
- Metrics MD5: `51a7c4a91c6efa46052f3fcbd6eecbc4`
- Accepted group: `A`
- Accepted split: `val`
- Seeds: `1, 2, 3`
- Scenarios: `clean, rgb_missing, ir_missing, rgb_noise_005, rgb_noise_010, ir_noise_005, ir_noise_010, rgb_blur_k7, ir_blur_k7, both_noise_005, both_blur_k7`
- Validated rows: `33`
- Generated at: `2026-09-10T01:11:18Z`
