# SXX — study title

Review date and author; status (data complete / partial; review draft / complete).

## Motivation and hypotheses

State the original scientific question and falsifiable comparisons. Link the
study prescription; distinguish planned tests from post-hoc diagnostics.

## Design and controls

List architectures, optimizer, budgets, seeds, controlled variables, pairing,
noise and nuisance policies. Identify shared reference cohorts explicitly.

## Provenance and completeness

Record cluster/root, import namespace, sync date, source revision/archive and
hashes, dirty-tree limitations, expected/observed runs, final/best epochs and
stopping reasons. Link local inventory and missing-artifact list.

## Evaluation contract

Record dataset/split/validation identities, evaluation counts and parameter
order, scaler/coordinate basis, held-out nuisance definitions and test flag.
Explain comparability limits; same nuisance within a pair is different from
held-out nuisance populations. Check zero-correction baselines and matching
ordered pair IDs before paired comparisons.

## Quantitative findings

Report seed-level results, means and sample SD, paired differences and relevant
slices/distance bins. Link generated CSVs and figures with input paths and exact
commands or lightweight notebook/code needed to reproduce them.

## Interpretation and limitations

Separate measurement from inference. Distinguish physical units, scaled-z
errors, Fisher sensitivity correlations and residual covariance diagnostics.
Do not imply downstream ADORA capture or identifiability from single-step skill.

## Follow-up and tracking

List unresolved identity questions and proposed controlled follow-up experiments.
Link the ML study index and tracker. State explicitly whether any untouched test
set was evaluated. Use [S06](s06/review.md) as a completed example and its
[reconciliation](s06/reconciliation.md) for provenance reporting.
