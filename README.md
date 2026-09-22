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

## Week 8 — Final Integration & Presentation

**Status:** `NoShowModelInterfaceV2` is final and demonstration-ready. It has
been tested against two internally-built models (LogisticRegression,
RandomForest) and, separately, against a model built on the Data Science
track's full confirmed methodology (including their `booking_month`
feature) — proving it can host a differently-featured model without
rework.

**Cross-track (Data Science):** Alia Al-Qadri's methodology has been
tested and validated end-to-end. Her `booking_month` feature was tested
directly and is **not adopted** (no ROC-AUC improvement). A serialized
model artifact from her is still pending as of this submission — the
interface is proven ready to receive it as soon as it arrives, with no
rework required.

**Known limitations carried into any future work:**
- No real fitted Data Science model artifact integrated yet — only a
  tested reproduction of Alia's confirmed methodology.
- No fairness evaluation has been run on the model's use of gender/age.
- The Sunday-appointment / Knowledge Base inconsistency from Week 5
  remains unresolved.

**Repository:** `interface.py` contains the final model-integration
contract. See the Week 7 and Week 8 reports (in this repo / linked in
project docs) for full testing evidence and results.
