# S09 analysis

The authoritative [scientific review](review.md) covers the completed nine-run V4
distance-balanced radial capture campaign: the radial mixture-ratio comparison,
the distance curriculum and its exposure confound, family-stratified results, and
the corrected per-parameter metrics.

The completed export `s09_complete_20261030T201514Z.tar.gz` was hash-verified
against its manifest (45/45) and found **byte-identical** to the pre-existing local
import, so nothing was copied or overwritten; provenance details are in the review.

An earlier per-parameter interpretation of S09 was wrong — it read the
alphabetically sorted `metrics.json` keys positionally. The regenerated outputs and
the changed values are recorded in
[`Results/ml_analysis/S09/SUPERSEDED.md`](../../../../../Results/ml_analysis/S09/SUPERSEDED.md).

Regenerate every table, figure and headline number with:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 \
  Results/ml_analysis/S09/s09_capture_range_analysis.ipynb
```

Shared loading and plotting examples are in the [ML study index and workflow](../README.md).
The [study prescription](../../s09/README.md) remains unchanged.
