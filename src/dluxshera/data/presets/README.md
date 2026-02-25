# SHERA preset catalog (Phase 8A strict nested schema)

Preset files live in `src/dluxshera/data/presets/` and are selected by name via:

```yaml
system:
  preset: SHERA_TESTBED_3P
```

The resolver loads the matching YAML/JSON file, then deep-merges user config on top.
Lists are replaced wholesale; mappings are merged recursively.

## Available presets

- `SHERA_TESTBED_3P` → `SHERA_TESTBED_3P.yaml`
- `SHERA_FLIGHT_3P` → `SHERA_FLIGHT_3P.yaml`
- `SHERA_TESTBED_2P` → `SHERA_TESTBED_2P.yaml`
- `SHERA_FLIGHT_2P` → `SHERA_FLIGHT_2P.yaml`

## 2P vs 3P

- `*_3P` presets set `system.optics.kind: three_plane` and carry three-plane geometry fields.
- `*_2P` presets set `system.optics.kind: two_plane` and carry two-plane geometry fields.

## Notes on legacy parity

These presets preserve numerical values from the in-code point-design configs in:

- `src/dluxshera/systems/three_plane.py` (`SHERA_TESTBED_CONFIG`, `SHERA_FLIGHT_CONFIG`)
- `src/dluxshera/systems/two_plane.py` (`SHERA_TESTBED_CONFIG`, `SHERA_FLIGHT_CONFIG`)

Detector defaults mirror the legacy builder baseline order:
`downsample` → `pixel_offsets` → `pixel_response` → `jitter`.

Some optics fields are included for parity even if current strict resolver validation only enforces a minimal subset.
Unknown keys are currently warning-only in resolver validation and are passed through for downstream dataclass conversion.
