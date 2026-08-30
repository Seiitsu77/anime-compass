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

## The production artifact

The evaluation artifact above is trained on the split, which withholds each
user's held-out positives. That is correct for measuring and wrong for serving.
A second artifact is built from every positive rating using the **identical**
frozen hyperparameters.

| | Evaluation | Production |
|---|---|---|
| File | `als_train_only.npz` | `data/processed/als_production_item_factors.npz` |
| `artifact_role` | `evaluation` | `production` |
| Rows scanned | 57,633,278 | 57,633,278 |
| Positive interactions | 24,916,911 | **30,875,410** (+23.9%) |
| Users | 308,177 | 308,177 |
| Withholds held-out positives | yes | no |
| Valid for holdout metrics | **yes** | **no** |
| SHA-256 | `a0be5f3f1dde0a40...` | `95c079b1b8f4e0e5...` |
| Size on disk | 7.0 MB | 7.1 MB |
| Build time | 268 s | 312 s |

The production artifact additionally pins `ratings_sha256`
(`b60519348a90bd5e...`) and `catalog_ids_sha256`
(`0ab8367a4c8a10a8...`), and carries `not_valid_for_holdout_evaluation: true`.

**The published holdout metrics must never be recomputed against the production
artifact.** Any holdout scored against it would consist of interactions it
already trained on. The serving loader enforces this in both directions by
checking `artifact_role`, so the two cannot be silently swapped.

Rebuild and verify with:

```powershell
python scripts/build_production_als.py
python scripts/verify_production_als.py
```

## Threshold variants

Thresholds 7 and 9 require their own splits and therefore their own ALS
artifacts, because the training positives differ by definition. Those artifacts
use **identical hyperparameters** to the table above — only the split changes.
Retuning per threshold would confound "is ALS robust to the relevance
definition" with "can ALS be retuned for each definition".
