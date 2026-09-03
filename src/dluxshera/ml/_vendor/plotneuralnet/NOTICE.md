# PlotNeuralNet vendored resources

This directory contains a minimal vendored subset of PlotNeuralNet:

- `layers/Ball.sty`
- `layers/Box.sty`
- `layers/RightBandedBox.sty`
- `layers/init.tex`

Upstream project: <https://github.com/HarisIqbal88/PlotNeuralNet>

Source commit: `e96bc852189c2089dd500527a0a01a5a36e8977e`

License: MIT. The upstream license text is preserved in `LICENSE`.

The Python renderer in `dluxshera.ml.visualization` is dLuxShera-authored
wrapper code. It does not call PlotNeuralNet's `tikzmake.sh` or launch a PDF
viewer.
