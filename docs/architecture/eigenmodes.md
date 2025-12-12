# Eigenmodes and EigenThetaMap

## What EigenThetaMap provides
- Computes and stores eigenmodes of the Fisher information / curvature matrix for a given θ reference point.
- Supplies linear transforms between θ-space and eigen-θ (`z`) space (`theta_from_z` and `z_from_theta`), supporting whitening and truncation.
- Keeps shapes and ordering explicit so optimisation code can treat eigen-coordinates as drop-in replacements.

## When to use eigenmodes
- Standard optimisation runs directly in θ-space; eigenmodes are an advanced option to precondition or reduce dimensionality.
- Canonical demos note this pathway: compute curvature once, build `EigenThetaMap`, and optimise in z-space while reusing the same Binder-based loss.
