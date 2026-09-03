# ML model architecture visualization

## Purpose

dLuxShera provides a lightweight architecture-visualization helper for ML
notebooks that need to inspect candidate model structure during development.
The initial use case is the shared-encoder pairwise correction CNN used by the
S01 baseline notebook.

This is a model-development diagnostic, not part of ML training or inference.
Training jobs, HPC runs, and ordinary package imports do not need LaTeX.

## Current backend

The current backend emits PlotNeuralNet-style TikZ/LaTeX. PlotNeuralNet was
chosen because its compact 3-D CNN boxes make convolutional research figures
readable at notebook and paper width.

The packaged `kgruiz/PlotNeuralNet` fork was tested first, but pip installation
failed at commit `ee1fc72c44ee210962830bcc6230258187e5e5b1` because the
repository's `pyproject.toml` is invalid TOML. dLuxShera therefore vendors only
the minimal MIT-licensed upstream PlotNeuralNet layer resources required for
rendering:

- upstream: <https://github.com/HarisIqbal88/PlotNeuralNet>
- source commit: `e96bc852189c2089dd500527a0a01a5a36e8977e`
- vendored location: `src/dluxshera/ml/_vendor/plotneuralnet/`

dLuxShera owns the Python rendering wrapper. It does not call PlotNeuralNet's
`tikzmake.sh` and never launches a PDF viewer.

## Architecture

The implementation deliberately separates model inspection from rendering:

```text
PyTorch model
    -> dLuxShera architecture description
    -> PlotNeuralNet/TikZ source
    -> .tex
    -> optional pdflatex
    -> .pdf
```

The architecture description records the convolutional stages, adaptive pooling
shape, projection layers, embedding dimension, comparator semantics, regression
head dimensions, output dimension, and trainable parameter count extracted from
the actual model instance.

## Dependencies

### Python dependency

No external Python PlotNeuralNet package is required. The small TikZ layer
resource subset is included as package data in dLuxShera.

For pairwise CNN visualization, install the ML Python extra because the helper
inspects a PyTorch model:

```bash
python -m pip install -e ".[ml,notebooks]"
```

### System dependency

PDF compilation requires an external LaTeX toolchain that provides `pdflatex`,
TikZ, `standalone`, and `import`.

High-level install options:

- macOS: MacTeX, or BasicTeX with the required TikZ/standalone packages.
- Linux: TeX Live with the required LaTeX/TikZ packages.
- Windows: MiKTeX or TeX Live.

If `pdflatex` is unavailable, call the helper with `compile_pdf=False` to write
the `.tex` source and JSON metadata without compiling a PDF.

The renderer resolves `pdflatex` by checking `PATH` first. On macOS, if that
fails, it also checks the standard MacTeX executable path:
`/Library/TeX/texbin/pdflatex`. This handles GUI-launched Jupyter environments
that do not inherit the same `PATH` as a terminal session. The library does not
modify `PATH` globally.

## Architecture views

dLuxShera intentionally emits two complementary views from the same extracted
architecture description.

### Model overview

Use the model overview when reasoning about:

- pairwise/Siamese topology;
- shared weights;
- comparator choice;
- prediction head.

This view hides individual convolution stages so the two images, single shared
encoder, embeddings, comparator, regression head, and `Delta theta` output are
readable at notebook or presentation width.

### Encoder detail

Use the encoder detail when reasoning about:

- convolution architecture;
- receptive/downsampling progression;
- channel counts;
- pooling;
- embedding dimension.

This view renders the encoder once. The pairwise A/B branches, comparator, and
regression head are intentionally omitted.

## Visual scaling

Box dimensions are presentation-scaled and are not linearly proportional to
tensor dimensions. Exact tensor shapes are shown explicitly in labels.

The renderer applies a deterministic bounded nonlinear mapping to spatial sizes
and channel counts. This preserves the visual ordering that spatial dimensions
decrease and channel depth increases, while preventing a `160 x 160` input plane
from dominating the page or a `4 x 4` pooled map from disappearing.

## Usage

```python
from dluxshera.ml.models import build_pairwise_correction_model
from dluxshera.ml.visualization import render_pairwise_correction_architecture_set

model_config = {
    "comparator": "concat_diff",
    "embedding_dim": 128,
    "adaptive_pool_shape": [4, 4],
}

model = build_pairwise_correction_model(science_dim, model_config)

architecture = render_pairwise_correction_architecture_set(
    model,
    input_shape=(1, 160, 160),
    model_config=model_config,
    output_dir="Results/ML Training Datasets/preprocessed/studies/visualizations/S01-E00",
    compile_pdf=True,
)

overview_pdf = architecture.overview.pdf_path
encoder_pdf = architecture.encoder.pdf_path
```

The two result objects report `tex_path`, `pdf_path`, `metadata_path`,
`backend`, and metadata. The older
`render_pairwise_correction_architecture(...)` helper remains available for
callers that expect one `ArchitectureRenderResult`; it now renders the model
overview view with the original basename default.

Reusable library code never imports IPython or displays the result. Notebook
display remains notebook-side code.

## Current scope

Version 1 supports the dLuxShera `PairwiseCorrectionCNN` family. It is not a
generic PyTorch graph renderer and should not be used as one.

The figure visibly tracks the configured comparator:

- `concat_diff`: `[h_A, h_B, h_B - h_A]` with dimension `3 * embedding_dim`.
- `difference`: `h_B - h_A` with dimension `embedding_dim`.

## Future extension

Possible future work, without changing the current renderer contract, includes:

- residual CNN encoders;
- multi-head outputs;
- nuisance-prediction heads;
- fixed physics/Fisher feature branches;
- latent or memory token architectures;
- alternative rendering backends.
