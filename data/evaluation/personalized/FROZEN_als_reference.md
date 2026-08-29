# Frozen ALS Reference Model

This configuration is **frozen** for the robustness phase. It is not retuned on
any confirmation or test sample. Every threshold, stratum, retrieval, and hybrid
experiment in this phase uses this exact model or a variant explicitly derived
from it.

## Selected configuration

| Parameter | Value |
|---|---|
| factors | 128 |
| alpha | 5.0 |
| regularization | 0.05 |
| iterations | 15 |
| cg_steps | 3 |
| seed | 42 |
| solver | Alternating least squares, conjugate gradient half-steps (Takács et al., 2011) |
| loss | Implicit-feedback weighted squared error (Hu, Koren, Volinsky, 2008) |
| confidence | `c_ui = 1 + alpha * r_ui`, binary `r_ui = 1` for train positives |

Selected by validation NDCG@10 over 12 candidates across two validation-only
rounds. The test split was not read during selection.

## Training matrix definition

- Rows: users with at least one train positive (308,177 of 310,059).
- Columns: all 18,064 catalog items.
- Entries: **train positives only** (`rating >= 8` and assigned to train).
  24,916,911 non-zero entries.
- Explicit negatives, neutral ratings, and ignored ratings are **not** entries.
  They are neither positives nor unobserved: they are excluded from candidates at
  ranking time as known items. No negative confidence is introduced.
- Held-out validation and test positives never appear in the matrix.

## Exported artifact

Item factors only, `(18064, 128) float32`, plus aligned `anime_ids (18064,) int64`.
User vectors are recomputed at request time by folding a user's positives into
item space:

```
x_u = (YᵀY + alpha · Y_uᵀY_u + reg · I)⁻¹ (alpha · Σ_{i∈u} y_i)
```

Verified faithful against the trained user factors on 2,000 sampled users:
cosine mean **0.9995** (min 0.9972, 100% above 0.95); top-20 ranking overlap mean
0.9728 (99.4% at or above 0.90).

## Reproducibility hashes

| Object | SHA-256 |
|---|---|
| Source ratings (`rating_complete.csv`) | `b60519348a90bd5e02c25355b374f7ca055a0637a237f0e163447953b13ffaa0` |
| Processed catalog | `2ef54a712f63eec2adc33f21bd431fc66ab097ba135ab440fd3d773f84668c75` |
| Catalog ID list | `64f84b07112b7132cbaac3ad2da25aa5a6b985a78b0fcfa379bf0b23bebdcde0` |
| Split (`holdout_seed42_pos8.sqlite`) | `a668114f043a54dc7048dddc8d5290416579b0eda5abbddc14bc47065c970038` |
| ALS artifact (`als_train_only.npz`) | `a0be5f3f1dde0a406d2bd14af705467a4b8155e8089a286e578c2f6f0ded354b` |

The artifact records its training split hash internally; the runner raises if it
does not match the split being evaluated.

## Build cost

268 s wall on CPU for 15 iterations over 308,177 users and 24.9M positive edges.
Serving artifact is 9.3 MB of float32.

## Threshold variants

Thresholds 7 and 9 require their own splits and therefore their own ALS
artifacts, because the training positives differ by definition. Those artifacts
use **identical hyperparameters** to the table above — only the split changes.
Retuning per threshold would confound "is ALS robust to the relevance
definition" with "can ALS be retuned for each definition".
