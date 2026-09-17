## Week 7 — Testing, Reliability & Refinement

**Tested:** training reproducibility, preprocessing edge cases (first-time
patients, missing `reminder_channel`, unseen categorical values), and the
model-integration interface's handling of invalid/unexpected input.

**Found:**
- The Week 6 interface (`NoShowModelInterface`) had no dtype validation —
  a malformed numeric value crashed with a raw scikit-learn error instead
  of a clear one.
- It also hardcoded one global feature set, so it couldn't host a model
  needing a different feature (proven by testing it against a model built
  on the Data Science track's `booking_month` feature).

**Fixed:** replaced with `NoShowModelInterfaceV2` (`src/models/interface.py`)
— per-model `feature_fn`/`feature_cols`/`numeric_cols`, dtype validation,
and logging. Both issues retested and confirmed resolved.

**Cross-track (Data Science):** Alia Al-Qadri's full methodology matches
this pipeline almost exactly, except for one feature, `booking_month`.
Tested directly: it does not improve ROC-AUC (0.682 vs. 0.683 baseline) and
is **not adopted** — documented as tested-and-rejected, same as Week 6's
tree-model comparison.

**Still open:** no real fitted Data Science model artifact yet; no fairness
evaluation on gender/age; Week 5's Sunday-appointment / Knowledge Base
inconsistency remains unresolved.
