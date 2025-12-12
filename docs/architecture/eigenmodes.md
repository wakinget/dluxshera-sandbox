# Eigenmodes and EigenThetaMap

## Why eigenmodes?
Some parameter combinations are far better constrained than others. Working in an eigenbasis of a local information metric (such as a Fisher Information Matrix) can highlight these directions, precondition optimisation, and make it easier to interpret which physical effects drive the fit.

## EigenThetaMap
`EigenThetaMap` captures the linear mapping between native θ-space and an eigen-space derived from a curvature estimate. It exposes helpers to map θ → z and z → θ, making whitening or truncating modes straightforward. The map plugs directly into the same Binder-based losses used elsewhere; only the parameter coordinates change.

## How eigenmodes fit into the pipeline
Eigenmodes are an optional layer on top of the standard inference pipeline. A typical flow computes a local FIM, builds an `EigenThetaMap`, runs gradient descent in eigen coordinates, and then maps the result back to θ. Canonical workflows keep everything else the same—the Binder produces images, loss helpers compare to data—and eigenmodes simply provide an alternative basis when it improves convergence or interpretation.
