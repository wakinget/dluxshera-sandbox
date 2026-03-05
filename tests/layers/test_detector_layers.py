import jax.numpy as jnp
import pytest

pytest.importorskip("interpax")

from dLux.psfs import PSF

from dluxshera.layers.detector_layers import ApplyPixelOffsets


def _gaussian_image(n: int = 17, sigma: float = 2.5):
    y, x = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing="ij")
    c = (n - 1) / 2
    return jnp.exp(-((x - c) ** 2 + (y - c) ** 2) / (2 * sigma**2))


def test_apply_pixel_offsets_init_validation():
    with pytest.raises(ValueError, match="dx_map must be a 2D array"):
        ApplyPixelOffsets(dx_map=jnp.zeros((4,)), dy_map=jnp.zeros((4, 4)))

    with pytest.raises(ValueError, match="dy_map must be a 2D array"):
        ApplyPixelOffsets(dx_map=jnp.zeros((4, 4)), dy_map=jnp.zeros((4,)))

    with pytest.raises(ValueError, match="must have the same shape"):
        ApplyPixelOffsets(dx_map=jnp.zeros((3, 4)), dy_map=jnp.zeros((4, 4)))

    with pytest.raises(ValueError, match="interp_method must be one of"):
        ApplyPixelOffsets(dx_map=jnp.zeros((4, 4)), dy_map=jnp.zeros((4, 4)), interp_method="bad")


def test_apply_pixel_offsets_shape_mismatch_raises():
    image = _gaussian_image(n=17)
    psf = PSF(data=image, pixel_scale=1.0)

    dx = jnp.zeros((16, 16))
    dy = jnp.zeros((16, 16))
    layer = ApplyPixelOffsets(dx_map=dx, dy_map=dy)

    with pytest.raises(ValueError, match="shape must match psf.data.shape"):
        layer.apply(psf)


@pytest.mark.parametrize("method", ["linear", "cubic"])
def test_apply_pixel_offsets_identity(method):
    image = _gaussian_image()
    psf = PSF(data=image, pixel_scale=1.0)

    zeros = jnp.zeros_like(image)
    layer = ApplyPixelOffsets(dx_map=zeros, dy_map=zeros, interp_method=method)

    out = layer.apply(psf)

    assert out.data.shape == image.shape
    assert jnp.allclose(out.data, image, atol=1e-5, rtol=1e-5)


def test_apply_pixel_offsets_boundary_clamping_finite_and_edge_like():
    image = jnp.arange(25.0).reshape(5, 5)
    psf = PSF(data=image, pixel_scale=1.0)

    dx = jnp.full_like(image, 100.0)
    dy = jnp.full_like(image, -100.0)

    layer = ApplyPixelOffsets(dx_map=dx, dy_map=dy, interp_method="linear")
    out = layer.apply(psf)

    expected = jnp.full_like(image, image[0, -1])
    assert jnp.isfinite(out.data).all()
    assert jnp.allclose(out.data, expected)


def test_apply_pixel_offsets_clip_nonnegative():
    image = _gaussian_image() - 2.0
    psf = PSF(data=image, pixel_scale=1.0)

    zeros = jnp.zeros_like(image)
    layer = ApplyPixelOffsets(
        dx_map=zeros,
        dy_map=zeros,
        interp_method="linear",
        clip_nonnegative=True,
    )

    out = layer.apply(psf)

    assert out.data.min() >= 0.0


def test_apply_pixel_offsets_fft_mode_clear_error_if_not_queryable(monkeypatch):
    image = _gaussian_image(n=9)
    psf = PSF(data=image, pixel_scale=1.0)
    zeros = jnp.zeros_like(image)

    layer = ApplyPixelOffsets(dx_map=zeros, dy_map=zeros, interp_method="fft")

    def _no_query_fft(values):
        return values

    monkeypatch.setattr("dluxshera.layers.detector_layers.interpax.fft_interp2d", _no_query_fft)

    with pytest.raises(ValueError, match="not supported for per-pixel offsets"):
        layer.apply(psf)
