# Lessons Learned

- 2025-12-18 — Zodiax dotted-key filtering vs params containers:
  - **Symptom:** When ModelParams/SheraThreePlaneParams store external parameter names with dots (e.g., `"m1_aperture.coefficients"`), passing those dotted strings to `zdx.filter_value_and_grad` makes Zodiax treat them as structural paths, triggering missing-attribute errors during pure-mode optimization.
  - **Failed attempt:** Supplying tuple paths (e.g., `( "params", "m1_aperture.coefficients" )`) to avoid dot splitting causes `hasattr`/attribute lookups on tuples inside Zodiax's filter helpers, resulting in `TypeError: attribute name must be string`.
  - **Resolution:** For gradients over params containers, bypass Zodiax filtering and call `jax.value_and_grad` directly on the params dictionary, letting `eqx.filter_jit` mask nondifferentiable leaves. Reserve `zdx.filter_value_and_grad` for model-object gradients where dotted paths intentionally traverse the model tree.
