# S10 in-flight import reconciliation — 2026-11-02

The completed nine-run collection is installed under
`Results/hpc_imports/ml/gattaca2/S10/`. All 47 files were independently hashed
against `Results/ml_analysis/S10/import_stage_20261102/`; every file matched.
The export manifest's per-file inventory also matched actual installed hashes.
All nine histories contain exactly epochs 0–599, and all manifests explicitly
record test_evaluated=false.

Archive `s10_complete_20261102T180701Z.tar.gz` hashes to
`e29a7ee1d1595652a95bf25503e9f7a4e8f1eebde224e29bcefc610d5435d226`.
The archive and normalized staging copy remain unchanged.

The prior eight-run population is preserved at
`Results/hpc_imports/ml_history/gattaca2/S10_inflight_preserved_20261102/`, outside
active recursive discovery. The supplied import account reports three already
final runs, five stale/incomplete runs and one absent run; detailed historical
per-run comparison remains to be independently documented. This reconciles an
in-flight snapshot, not an unrelated training generation.

Transfer was by SCP archive, followed by local promotion; no rsync operation is
claimed. The new local archive-import audit record is
`Results/ml_analysis/S10/closeout/archive_import_audit.json`, with archive hash,
installed inventory and exported run provenance. S11 has an analogous record
for its six new runs and 31 files. These are local audit records, not fabricated
remote transfer logs.

Source revision `0f20f63ada40677d12068bfc89d90d6be1bdd942` has dirty=true throughout;
without the working-tree diff the commit cannot reproduce the executed source.
No remote filesystem or scheduler was queried during this reconciliation.
