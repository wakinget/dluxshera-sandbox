import jax.numpy as jnp
import pytest

from dLux.psfs import PSF

from dluxshera.layers.detector_layers import ApplyPixelOffsets


def _gaussian_image(n: int = 17, sigma: float = 2.5):
    y, x = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing="ij")
    c = (n - 1) / 2
    return jnp.exp(-((x - c) ** 2 + (y - c) ** 2) / (2 * sigma**2))


def test_apply_pixel_offsets_identity():
    image = _gaussian_image()
    psf = PSF(data=image, pixel_scale=1.0)

    zeros = jnp.zeros_like(image)
    layer = ApplyPixelOffsets(dx_map=zeros, dy_map=zeros)

    out = layer.apply(psf)

    assert jnp.allclose(out.data, image)


def test_apply_pixel_offsets_shape_mismatch_raises():
    image = _gaussian_image(n=17)
    psf = PSF(data=image, pixel_scale=1.0)

    dx = jnp.zeros((16, 16))
    dy = jnp.zeros((16, 16))
    layer = ApplyPixelOffsets(dx_map=dx, dy_map=dy)

    with pytest.raises(ValueError, match="shape must match psf.data.shape"):
        layer.apply(psf)
