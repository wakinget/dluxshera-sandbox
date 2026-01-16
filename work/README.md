# `work/` — Local working area

This directory is a home for **work-in-progress code** and supporting artifacts that don’t yet belong in `src/`, `tests/`, or `examples/`.

Nothing in `work/` is considered stable API or a curated example. Treat it as a lab bench: useful, flexible, and allowed to be messy.

---

## Purpose

Use `work/` for:

- Rapid prototyping and exploration
- One-off scripts used to answer specific questions
- Porting and preserving legacy scripts for reference
- Scratch notebooks/scripts before promoting them elsewhere

---

## Directory layout

### `work/experiments/`

Longer-running investigations that may evolve over days/weeks.

Suggested contents:
- Focused experiments (e.g., parameter sweeps, inference diagnostics)
- Small helper modules local to the experiment
- Notes / log files describing findings and decisions

Recommended structure:
    work/experiments/<topic_or_date>/
        README.md
        run_*.py
        notes.md
        outputs/            (ignored or ephemeral)
        data/               (ignored or external)

### `work/scratch/`

Short-lived “try it and toss it” scripts.

Suggested contents:
- Tiny repro snippets
- Debug scripts
- Small proof-of-concept plots
- Quick sanity checks

Recommendation: keep filenames descriptive and disposable (e.g., `scratch_fim_shapes.py`).

### `work/legacy/`

A safe landing zone for scripts copied from older repos or historical workflows.

Suggested contents:
- Unmodified legacy scripts (or minimally patched to run)
- Reference implementations and old plotting utilities
- Notes explaining what the script did, and whether it’s still relevant

Recommendation:
- Prefer keeping legacy scripts **as-is**
- Add a short header comment at the top describing origin + context

---

## Promotion guidelines

As scripts mature, consider promoting them:

- `work/` → `examples/scripts/`  
  When a script becomes a **repeatable, user-facing example** of how to use the library.

- `work/` → `tests/`  
  When a script uncovers a **behavioral contract** that should be protected by a test.

- `work/` → `src/`  
  When code is reusable, documented, and worth maintaining as part of the package.

---

## Safety and version control notes

- `git pull` does **not** delete untracked or ignored files in `work/`.
- Avoid destructive commands like:
    git clean -xfd
  unless you are sure you want to delete untracked + ignored files.

If you have work you’d be sad to lose, commit it (even as WIP) or back it up elsewhere.

---

## Conventions (optional, but recommended)

- Put new work in a subfolder (e.g., `experiments/<topic>/`) rather than at the top level.
- Add a short `README.md` inside each experiment folder describing:
  - Goal / question
  - How to run
  - Current status / findings
- Prefer `if __name__ == "__main__":` guards for scripts that may touch multiprocessing or hardware.
