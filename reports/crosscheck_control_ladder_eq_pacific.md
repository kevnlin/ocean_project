# Cross-check — P0 (eq_pacific)

> **Track B is pending.** Track A ran; the intern's independent run has not been supplied. The comparison columns below are placeholders, and no agreement should be inferred from them.

> **What P3 cross-checks, and what it cannot.** The ladder's rows (`thin_*`, `superob_*`) are P0 rows, so the numbers below are the P0 comparison restricted to them. The plan also asks the two tracks to agree on the thinning budget, grid definition, merge radius, noise weighting and modality handling: both tracks instantiate the ladder from the same `godas_model.build_row` registry and load the same checkpoints, so those five are shared **by construction, not by agreement** — two independent implementations of them would be a stronger check and do not exist. What is independently checked is the scoring of the rows they produce.

## Provenance

| field | Track A | Track B |
|---|---|---|
| protocol version | `—` | `pending` |
| protocol hash | `—` | `pending` |
| git commit | `—` | `pending` |
| git dirty | `—` | `pending` |
| seeds | `—` | `pending` |
| created (UTC) | `—` | `pending` |
