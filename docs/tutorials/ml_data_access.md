# SHERA ML data access

This tutorial is for collaborators who want to use prepared SHERA ML data with
their own notebooks and models.  The reusable data layer can be used
independently of the dLuxShera optical model, independently of the current
pairwise CNN training code, and independently of the experiment campaign
machinery.

The goal is practical data plumbing: inspect a prepared dataset, read sharded
images efficiently, select samples by science and nuisance identity, form
controlled image differences, optionally sample pairs, and avoid obvious
train/test leakage.

## Install

From the repository root, create an environment and install the package in
editable mode.  The ``notebooks`` extra installs Jupyter support; the ``ml``
extra installs optional PyTorch helpers.

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[notebooks,ml]"
```

If you only need catalog and sharded-array access, PyTorch is not required:

```bash
python -m pip install -e ".[notebooks]"
```

## Prepared data handoff

The Git repository contains code, tests, notebooks, and metadata conventions;
large prepared image datasets are external data products and are not expected
to live in Git.  A collaborator needs access to a complete prepared dataset
directory, then should point ``PREPARED_ROOT`` at that directory itself, not at
its ``shards/`` subdirectory.

At a high level, the prepared root must contain:

```text
prepared_dataset/
  manifest.json
  vector_spaces.json
  index.jsonl
  array_shards_manifest.json
  shards/
```

Validate the handoff before building notebook logic around it:

```python
from pathlib import Path

from dluxshera.ml import load_sample_catalog

prepared_root = Path("/path/to/prepared_dataset")
catalog = load_sample_catalog(prepared_root)
catalog.summary()
```

## Core concepts

A **prepared dataset** is a directory optimized for ML access.  It contains
small metadata files plus sharded ``.npy`` image arrays:

```text
prepared_dataset/
  manifest.json
  vector_spaces.json
  index.jsonl
  array_shards_manifest.json
  shards/
    shard_00000.npy
    shard_00001.npy
```

A **sample** is one rendered image and its metadata row.  Each sample has a
stable ``sample_id``, a catalog row index for metadata arrays, and an
``array_index`` for image access.

A **science state/group** identifies the physical perturbation state.  All
nuisance realizations of the same physical state share the same science group.

A **nuisance state/group** identifies a registration/rendering nuisance
realization.  Different science states can be evaluated under the same nuisance
group.

Keeping science and nuisance identities separate is what lets you ask questions
such as "all nuisance realizations of this one science state" or "several
science states under the same nuisance realization."

A **pair** is an ordered comparison between two samples, A and B.  Pair utilities
are optional; direct catalog and image access is enough for many notebooks.

## Load a prepared catalog

```python
from pathlib import Path

from dluxshera.ml import load_sample_catalog

prepared_root = Path("/path/to/prepared_dataset")
catalog = load_sample_catalog(prepared_root)
```

The catalog is the high-level entry point for metadata and sample selection.  It
does not load the image shards into memory.

## Inspect the dataset

```python
catalog.summary()
```

Useful fields include:

```python
catalog.sample_count
catalog.sample_shape
catalog.science_dim
catalog.nuisance_dim
catalog.science_group_count
catalog.nuisance_group_count
catalog.parameter_labels
catalog.nuisance_labels
```

Inspect one sample:

```python
row = 0
catalog.sample_metadata(row)
```

The metadata record includes ``catalog_index`` for catalog arrays and
``array_index`` for sharded image reads.

## Read images efficiently

Use the catalog to open an ``ArrayShardReader``.  Shards are memory mapped and
cached lazily, so this does not load the full dataset into RAM.

```python
row = 0
array_index = int(catalog.array_indices[row])

with catalog.image_reader(cache_size=4) as reader:
    image = reader[array_index]
```

``reader[index]`` and ``reader.get(index)`` return independent array copies by
default.  ``reader.get(index, copy=False)`` returns a short-lived view into the
cached memory map; only use that when you know the view will not outlive cache
eviction or ``reader.close()``.

## Select controlled subsets

All nuisance realizations for one science state:

```python
science_id = catalog.science_groups[0]
rows = catalog.indices_for_groups(science_groups=[science_id])

for row in rows[:5]:
    print(catalog.sample_metadata(int(row)))
```

Multiple science states under one fixed nuisance realization:

```python
nuisance_id = catalog.nuisance_groups[0]
rows = catalog.indices_for_groups(nuisance_groups=[nuisance_id])

for row in rows[:5]:
    print(
        catalog.sample_ids[row],
        catalog.science_group_ids[row],
        catalog.nuisance_group_ids[row],
    )
```

Retrieve the corresponding images:

```python
array_indices = catalog.array_indices_for_groups(
    science_groups=[science_id],
    nuisance_groups=[nuisance_id],
)

with catalog.image_reader() as reader:
    images = [reader[int(array_index)] for array_index in array_indices[:4]]
```

## Build a difference image

For a fixed nuisance realization, choose two different science states:

```python
nuisance_id = catalog.nuisance_groups[0]
rows = catalog.indices_for_groups(nuisance_groups=[nuisance_id])

row_a = int(rows[0])
row_b = next(
    int(row)
    for row in rows[1:]
    if catalog.science_group_ids[row] != catalog.science_group_ids[row_a]
)

with catalog.image_reader() as reader:
    image_a = reader[int(catalog.array_indices[row_a])]
    image_b = reader[int(catalog.array_indices[row_b])]

difference = image_b - image_a
target_delta_z = catalog.fisher_scaled_deltas[row_b] - catalog.fisher_scaled_deltas[row_a]
```

This example does not prescribe a classifier or regression model.  It only shows
how to obtain a controlled image comparison and the corresponding science-vector
difference.

## Controlled pair sampling

``PairSampler`` draws ordered comparisons without constructing an O(N^2) table
of every possible pair.

```python
import numpy as np

from dluxshera.ml import PairPolicy, PairSampler, generate_split_registry

pair_demo_registry = generate_split_registry(
    catalog,
    seed=11,
    science_fractions={"train": 1.0},
    nuisance_fractions={"train": 1.0},
)
policy = PairPolicy(
    family_weights={"same_nuisance_different_science": 1.0},
    min_fisher_distance=0.0,
)
sampler = PairSampler(catalog, pair_demo_registry, policy)

record = sampler.sample_pair(np.random.default_rng(0))

with catalog.image_reader() as reader:
    image_a = reader[record.sample_a_index]
    image_b = reader[record.sample_b_index]

difference = image_b - image_a
```

Supported pair families include:

- ``same_nuisance_different_science``: same nuisance group, different science
  groups.
- ``different_nuisance_same_science``: same science group, different nuisance
  groups.  ``same_science_different_nuisance`` is accepted as an alias.
- ``different_science_different_nuisance``: both science and nuisance groups
  differ.
- ``identity``: a sample compared to itself, only when explicitly enabled.

Legacy aliases ``A``, ``B``, ``C``, and ``I`` map to those families.

Pair records use ordered target semantics: ``target_delta_z = z_B - z_A``.  For
same-science nuisance comparisons, the science target is zero and
``nuisance_delta`` records the nuisance-vector difference.

## Reproducible splits

``SplitRegistry`` separates science-state splits from nuisance-state splits.
This matters because each axis controls a different leakage question.

```python
from dluxshera.ml import generate_split_registry

split_registry = generate_split_registry(
    catalog,
    seed=11,
    science_fractions={"train": 0.8, "validation": 0.1, "test": 0.1},
    nuisance_fractions={"train": 0.8, "validation": 0.1, "test": 0.1},
    require_nonempty_nuisance_partitions=catalog.nuisance_group_count >= 3,
)

train_science = split_registry.science_groups("train")
validation_science = split_registry.science_groups("validation")
train_nuisance = split_registry.nuisance_groups("train")
```

For held-out science states under seen nuisance realizations, ask the sampler
for ``science_split="validation"`` and ``nuisance_split="train"``.  For a
stricter held-out-science and held-out-nuisance evaluation, use
``science_split="validation"`` and ``nuisance_split="validation"``.

If a prepared dataset contains only one or two nuisance realizations, nonempty
validation and test nuisance splits are impossible.  Keep nuisance groups in
``"train"`` for pair sampling, or set
``require_nonempty_nuisance_partitions=False`` for split inspection until a full
nuisance bank is available.

Persist split registries when results need to be reproduced:

```python
from dluxshera.ml import load_split_registry, write_split_registry

write_split_registry(Path("split_registry.json"), split_registry)
split_registry = load_split_registry(Path("split_registry.json"), catalog=catalog)
```

## Optional PyTorch integration

The tutorial above does not require PyTorch.  If you want the repository to feed
pairs directly into a PyTorch training workflow, use:

```python
from dluxshera.ml.torch_data import DynamicPairDataset, PairManifestDataset
```

``DynamicPairDataset`` samples deterministic training pairs from a
``PairSampler``.  ``PairManifestDataset`` replays frozen validation/test pairs
from a pair manifest.

## Where to go deeper

- Prepared dataset infrastructure:
  [docs/dev/ml_prepared_dataset_wave1.md](../dev/ml_prepared_dataset_wave1.md)
- V3 training dataset workflow:
  [docs/dev/ml_training_dataset_v3.md](../dev/ml_training_dataset_v3.md)
- ML inverse-model design notes:
  [docs/dev/shera_ml_inverse_model_design.md](../dev/shera_ml_inverse_model_design.md)
- Current dated ML program status:
  [docs/dev/notes/ml_program_status_20260908.md](../dev/notes/ml_program_status_20260908.md)
