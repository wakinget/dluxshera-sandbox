# Lessons Learned

- 2026-05-22 - Keep packaging metadata in one canonical place:
  - **Symptom:** `pyproject.toml`, `requirements.txt`, and setup docs diverged,
    which made environment setup inconsistent across local and agent workflows.
  - **Resolution:** make `pyproject.toml` canonical, keep requirements files as
    compatibility shims, and document editable install (`-e ".[dev]"`) as the
    default setup path.

- 2026-05-22 - `/usr/bin/time` portability is not enough; capability probing is required:
  - **Symptom:** On BSD/macOS hosts, `/usr/bin/time` exists but rejects GNU
    `-v`, causing wrapper subprocess failures even when child commands are valid.
  - **Resolution:** Probe GNU timing support before enabling `-v`, then fall
    back to portable timing or disabled external timing while preserving child
    return-code semantics and recording requested/effective mode in diagnostics.

- 2026-05-22 - Observation summary units are not optimizer loss units:
  - **Symptom:** A recovered-reference solve may need `subblock_reduce: mean`,
    but consuming that optimizer-normalized Schur summary as observation
    information undercounts a multi-frame subblock.
  - **Resolution:** Default Schur export records summed-likelihood information
    accounting, and real-summary forecast/update consumers require that scale
    metadata unless a legacy/debug override is explicitly recorded.

- 2025-12-18 — Zodiax dotted-key filtering vs params containers:
  - **Symptom:** When ModelParams/SheraThreePlaneParams store external parameter names with dots (e.g., `"m1_aperture.coefficients"`), passing those dotted strings to `zdx.filter_value_and_grad` makes Zodiax treat them as structural paths, triggering missing-attribute errors during pure-mode optimization.
  - **Failed attempt:** Supplying tuple paths (e.g., `( "params", "m1_aperture.coefficients" )`) to avoid dot splitting causes `hasattr`/attribute lookups on tuples inside Zodiax's filter helpers, resulting in `TypeError: attribute name must be string`.
  - **Resolution:** For gradients over params containers, bypass Zodiax filtering and call `jax.value_and_grad` directly on the params dictionary, letting `eqx.filter_jit` mask nondifferentiable leaves. Reserve `zdx.filter_value_and_grad` for model-object gradients where dotted paths intentionally traverse the model tree.
