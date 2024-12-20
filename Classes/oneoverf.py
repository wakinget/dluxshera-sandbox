import jax.numpy as np
import jax.random as jr
from jax import Array
from jax.numpy.fft import fft2, ifft2, fftshift

import time

__all__ = ["oneoverf_noise_2D"]


def oneoverf_noise_2D(n_samps: int, alpha: float, oversample: int = 1, key=None, rmTipTilt: bool = True) -> Array:
    """
    Generate 1/f^alpha noise in 2 Dimensions using Fourier-space convolution.
    Author: Dylan McKeithen 2024-12-19
    Uses fourier-space convolution to filter a noise image
        with a 1/f^alpha power spectrum.  The oversampling factor
        is used to add low-spatial frequency information to the
        noise, and helps avoid edge effects.  An image of
        size (n_samps*oversample)^2 is generated and filtered, and the
        central n_samps*n_samps is extracted and returned.
        The output is normalized by setting the variance of the
        (n_samps*oversample)^2 image to 1

    Parameters
    ----------
    n_samps: int
        The linear number of samples desired in the output.
    alpha: float
        The PSD exponent.
    oversample: int = 1
        The multiplicative oversampling factor, >= 1
    key: Optional jax.random.key
        jax.random.key for reproducible noise maps, random key is generated if not provided
    rmTipTilt: bool = True
        flag to remove tip/tilt from the generated noise map. True by default

    """

    # Ensure oversample is >= 1
    oversample = max(1, oversample)

    N = n_samps * oversample

    if key is None: # Generate a random key
        seed = int(time.time())
        key = jr.PRNGKey(seed)

    # Generate noise from normal distribution
    raw_noise = jr.normal(key, (N, N))

    # Calculate the PSD
    x = (np.arange(N) - N / 2).reshape(-1, 1) / N
    y = (np.arange(N) - N / 2).reshape(1, -1) / N
    envelope = fftshift(1.0 / (x ** 2 + y ** 2) ** (alpha / 4))
    # envelope[0, 0] = 0  # Avoid division by zero at the origin
    envelope = envelope.at[0, 0].set(0)  # Set the origin value to 0

    # Convolve the noise with the PSD
    FT = fft2(raw_noise) * envelope
    filtered_noise = np.real(fftshift(ifft2(FT)))

    # Normalize to variance = 1.0
    filtered_noise = filtered_noise / np.std(filtered_noise)

    # Remove Tip/Tilt if requested
    if rmTipTilt:
        # Fit filtered_noise to a plane and remove it
        X, Y = np.meshgrid((np.arange(N) - N / 2) / N, (np.arange(N) - N / 2) / N)
        XX = X.flatten()
        YY = Y.flatten()

        A = np.c_[XX, YY, np.ones(XX.shape)]  # Z = a*X + b*Y + c
        C = np.dot(np.linalg.pinv(A), filtered_noise.flatten())  # Least-squares solution
        fitted_plane = np.dot(A, C).reshape(filtered_noise.shape)
        filtered_noise -= fitted_plane

    # Extract the central portion of the image
    start = N // 2 - n_samps // 2
    end = start + n_samps
    central_noise = filtered_noise[start:end, start:end]

    return central_noise