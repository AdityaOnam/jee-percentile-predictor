# Data

Every file references a `source_id` in [`sources.json`](sources.json).

```text
data/
├── sources.json              provenance: URL, coverage and reliability of every source
├── percentiles/              published marks → percentile tables
│   ├── jee-main-2026-january-percentiles.json     10 shifts · 23 tiers (90 – 99.9)
│   ├── jee-main-2025-january-percentiles.json     10 shifts · 14 tiers · cross-validated (Vedantu = PW)
│   ├── jee-main-2025-april-percentiles.json        9 shifts · 13 tiers (92 – 99) → 90/91 imputed
│   ├── jee-main-2023-january-percentiles.json     12 shifts · Stage-1 reference only
│   └── jee-main-2022-national-percentile-curve.json  national binned curve, reference only
├── difficulty_overall/       per-shift Easy / Moderate / Hard counts (all subjects)
└── difficulty_subject/       per-shift, per-subject Easy / Moderate / Hard counts
```

## Schemas

**Percentile curve** (`percentiles/*.json` → `shifts[]`)

| Field | Example |
|---|---|
| `shift_id` | `2025_22_JAN_1` (year, day, month, shift) |
| `date`, `session_time` | `2025-01-22`, `Morning` |
| `percentile_curve[]` | `{ "percentile": 99.0, "raw_score": 186 }` (raw score out of 300) |

**Difficulty** (`difficulty_subject/*.json` → `shifts[]`)

| Field | Example |
|---|---|
| `subjects.physics` | `{ "total": 25, "n_easy": 9, "n_moderate": 16, "n_hard": 0 }` |

## How difficulty was measured

Every question of the actual paper was tagged Easy / Moderate / Hard in a hand-maintained question
bank, and the tags were counted per shift and subject. This is a single-annotator labelling, so treat it
as a consistent relative measure rather than ground truth.

## Quality notes

1. **2026 subject labels are noisier than 2025's.** 14 subject-shift combinations have questions filed under the wrong subject (see each file's `caveat`). 2025 is clean.
2. **2023 has no difficulty labels**, so its curves are fit for reference (Stage 1) and excluded from training.
3. **April 2025 starts at the 92nd percentile.** The 90th and 91st tiers are imputed by inverting each shift's own Normal-CDF fit (`jee_percentile.data.load_targets`). `n_imputed` marks those rows.
