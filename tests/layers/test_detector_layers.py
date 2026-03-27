import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("interpax")
import interpax

from dLux.psfs import PSF
from dLux.layers.detector_layers import Downsample

from dluxshera.builders.detector import build_detector_layer
from dluxshera.layers.detector_layers import ApplyConvolution, ApplyPixelOffsets


def _gaussian_image(n: int = 17, sigma: float = 2.5):
    y, x = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing="ij")
    c = (n - 1) / 2
    return jnp.exp(-((x - c) ** 2 + (y - c) ** 2) / (2 * sigma**2))


def _delta_psf(n: int = 41) -> PSF:
    data = jnp.zeros((n, n), dtype=float)
    center = n // 2
    data = data.at[center, center].set(1.0)
    return PSF(data=data, pixel_scale=1.0)


def _weighted_moments(image: jnp.ndarray) -> tuple[float, float, float]:
    y, x = jnp.meshgrid(
        jnp.arange(image.shape[0], dtype=float),
        jnp.arange(image.shape[1], dtype=float),
        indexing="ij",
    )
    total = jnp.sum(image)
    x_mean = jnp.sum(image * x) / total
    y_mean = jnp.sum(image * y) / total
    dx = x - x_mean
    dy = y - y_mean
    var_x = jnp.sum(image * dx * dx) / total
    var_y = jnp.sum(image * dy * dy) / total
    cov_xy = jnp.sum(image * dx * dy) / total
    return float(var_x), float(var_y), float(cov_xy)


def _principal_axis_angle_deg(image: jnp.ndarray) -> float:
    var_x, var_y, cov_xy = _weighted_moments(image)
    return float(0.5 * jnp.rad2deg(jnp.arctan2(2.0 * cov_xy, var_x - var_y)))


def _projected_variance(image: jnp.ndarray, theta_deg: float) -> float:
    var_x, var_y, cov_xy = _weighted_moments(image)
    theta = jnp.deg2rad(theta_deg)
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return float(c * c * var_x + s * s * var_y + 2.0 * c * s * cov_xy)


def _shift_image_linear(image: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    height, width = image.shape
    y, x = jnp.meshgrid(
        jnp.arange(height, dtype=float),
        jnp.arange(width, dtype=float),
        indexing="ij",
    )
    xq = jnp.clip(x - dx, 0.0, width - 1.0)
    yq = jnp.clip(y - dy, 0.0, height - 1.0)
    f = image.T
    shifted_flat = interpax.interp2d(
        xq.reshape(-1),
        yq.reshape(-1),
        jnp.arange(width, dtype=float),
        jnp.arange(height, dtype=float),
        f,
        method="linear",
        extrap=False,
    )
    return shifted_flat.reshape(height, width)


def test_apply_convolution_preserves_shape_and_flux_for_well_padded_psf():
    psf = _delta_psf(n=51)
    layer = ApplyConvolution(
        kernel_kind="gaussian",
        sigma_x=2.0,
        sigma_y=1.5,
        theta_deg=0.0,
        kernel_size=11,
        units="psf_pix",
    )

    out = layer.apply(psf)

    assert out.data.shape == psf.data.shape
    assert jnp.isclose(jnp.sum(out.data), 1.0, atol=1e-6, rtol=1e-6)


def test_apply_convolution_axis_aligned_anisotropic_gaussian_behaves_as_expected():
    psf = _delta_psf(n=51)
    layer = ApplyConvolution(
        kernel_kind="gaussian",
        sigma_x=3.0,
        sigma_y=1.0,
        theta_deg=0.0,
        kernel_size=17,
        units="psf_pix",
    )

    out = layer.apply(psf)
    var_x, var_y, cov_xy = _weighted_moments(out.data)

    assert abs(cov_xy) < 1e-6
    assert var_x > 2.0 * var_y


def test_apply_convolution_rotation_changes_anisotropic_blur_orientation():
    psf = _delta_psf(n=51)
    theta_zero = ApplyConvolution(
        kernel_kind="gaussian",
        sigma_x=3.0,
        sigma_y=1.0,
        theta_deg=0.0,
        kernel_size=17,
        units="psf_pix",
    )
    theta_rot = ApplyConvolution(
        kernel_kind="gaussian",
        sigma_x=3.0,
        sigma_y=1.0,
        theta_deg=45.0,
        kernel_size=17,
        units="psf_pix",
    )

    out_zero = theta_zero.apply(psf)
    out_rot = theta_rot.apply(psf)

    _, _, cov_zero = _weighted_moments(out_zero.data)
    _, _, cov_rot = _weighted_moments(out_rot.data)
    angle_rot = abs(_principal_axis_angle_deg(out_rot.data))

    assert abs(cov_zero) < 1e-6
    assert abs(cov_rot) > 0.1
    assert 30.0 < angle_rot < 60.0
    assert not jnp.allclose(out_zero.data, out_rot.data)


def test_apply_convolution_detector_pix_units_scale_to_psf_pixels():
    detector_units = ApplyConvolution(
        kernel_kind="gaussian",
        sigma_x=0.5,
        sigma_y=0.25,
        theta_deg=15.0,
        kernel_size=9,
        units="detector_pix",
        detector_to_psf_scale=3.0,
    )
    psf_units = ApplyConvolution(
        kernel_kind="gaussian",
        sigma_x=1.5,
        sigma_y=0.75,
        theta_deg=15.0,
        kernel_size=9,
        units="psf_pix",
    )

    assert jnp.allclose(
        detector_units.generate_kernel(),
        psf_units.generate_kernel(),
        atol=1e-6,
        rtol=1e-6,
    )


def test_apply_convolution_invalid_kernel_kind_raises():
    with pytest.raises(ValueError, match="kernel_kind values"):
        ApplyConvolution(
            kernel_kind="triangle",
            sigma_x=1.0,
            sigma_y=1.0,
            theta_deg=0.0,
            kernel_size=5,
            units="psf_pix",
        )


def test_apply_convolution_box_preserves_shape_and_flux_for_well_padded_psf():
    psf = _delta_psf(n=51)
    layer = ApplyConvolution(
        kernel_kind="box",
        width_x=2.0,
        width_y=4.0,
        kernel_size=13,
        units="psf_pix",
    )

    out = layer.apply(psf)

    assert out.data.shape == psf.data.shape
    assert jnp.isclose(jnp.sum(out.data), 1.0, atol=1e-6, rtol=1e-6)


def test_apply_convolution_box_symmetry_for_equal_widths():
    psf = _delta_psf(n=41)
    layer = ApplyConvolution(
        kernel_kind="box",
        width_x=5.0,
        width_y=5.0,
        kernel_size=11,
        units="psf_pix",
    )

    kernel = layer.generate_kernel()
    assert jnp.allclose(kernel, kernel.T, atol=1e-7, rtol=1e-7)


def test_apply_convolution_box_anisotropic_widths_change_axis_variance():
    psf = _delta_psf(n=51)
    layer = ApplyConvolution(
        kernel_kind="box",
        width_x=7.0,
        width_y=3.0,
        kernel_size=15,
        units="psf_pix",
    )

    out = layer.apply(psf)
    var_x, var_y, cov_xy = _weighted_moments(out.data)
    assert abs(cov_xy) < 1e-6
    assert var_x > 2.0 * var_y


def test_apply_convolution_box_detector_pix_units_scale_to_psf_pixels():
    detector_units = ApplyConvolution(
        kernel_kind="box",
        width_x=1.0,
        width_y=0.5,
        kernel_size=9,
        units="detector_pix",
        detector_to_psf_scale=3.0,
    )
    psf_units = ApplyConvolution(
        kernel_kind="box",
        width_x=3.0,
        width_y=1.5,
        kernel_size=9,
        units="psf_pix",
    )

    assert jnp.allclose(
        detector_units.generate_kernel(),
        psf_units.generate_kernel(),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize(("width_x", "width_y"), [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -1.0)])
def test_apply_convolution_box_invalid_widths_raise(width_x, width_y):
    with pytest.raises(ValueError, match="must be positive"):
        ApplyConvolution(
            kernel_kind="box",
            width_x=width_x,
            width_y=width_y,
            kernel_size=7,
            units="psf_pix",
        )


def test_apply_convolution_box_is_close_to_downsample_on_oversampled_grid():
    # A 4x oversampled detector pixel aperture is approximated by a 4-PSF-pixel box.
    # Exact equality is not expected because convolution+downsample and block-sum
    # downsample order of operations differ at finite support.
    n = 64
    oversample = 4
    image = _gaussian_image(n=n, sigma=5.0)
    image = image / jnp.sum(image)
    psf = PSF(data=image, pixel_scale=1.0)

    box_layer = ApplyConvolution(
        kernel_kind="box",
        width_x=1.0,
        width_y=1.0,
        kernel_size=11,
        units="detector_pix",
        detector_to_psf_scale=oversample,
    )
    conv_then_down = Downsample(oversample).apply(box_layer.apply(psf)).data
    down_only = Downsample(oversample).apply(psf).data

    assert conv_then_down.shape == down_only.shape
    assert jnp.isclose(jnp.sum(conv_then_down), jnp.sum(down_only), atol=1e-6, rtol=1e-6)
    assert jnp.allclose(conv_then_down, down_only, atol=2e-2, rtol=2e-2)


def test_apply_convolution_line_preserves_shape_and_flux_for_well_padded_psf():
    psf = _delta_psf(n=71)
    layer = ApplyConvolution(
        kernel_kind="line",
        length=7.0,
        theta_deg=25.0,
        sigma_perp=0.7,
        kernel_size=31,
        units="psf_pix",
    )

    out = layer.apply(psf)

    assert out.data.shape == psf.data.shape
    assert jnp.isclose(jnp.sum(out.data), 1.0, atol=1e-6, rtol=1e-6)


def test_apply_convolution_line_orientation_tracks_theta():
    psf = _delta_psf(n=71)
    layer = ApplyConvolution(
        kernel_kind="line",
        length=9.0,
        theta_deg=35.0,
        sigma_perp=0.6,
        kernel_size=33,
        units="psf_pix",
    )
    out = layer.apply(psf)
    angle = abs(_principal_axis_angle_deg(out.data))
    assert 20.0 < angle < 50.0


def test_apply_convolution_line_longer_length_increases_along_track_extent():
    psf = _delta_psf(n=71)
    theta = 30.0
    short = ApplyConvolution(
        kernel_kind="line",
        length=4.0,
        theta_deg=theta,
        sigma_perp=0.4,
        kernel_size=31,
        units="psf_pix",
    ).apply(psf)
    long = ApplyConvolution(
        kernel_kind="line",
        length=10.0,
        theta_deg=theta,
        sigma_perp=0.4,
        kernel_size=31,
        units="psf_pix",
    ).apply(psf)

    along_short = _projected_variance(short.data, theta)
    along_long = _projected_variance(long.data, theta)
    assert along_long > along_short


def test_apply_convolution_line_larger_sigma_perp_increases_cross_track_thickness():
    psf = _delta_psf(n=71)
    theta = 12.0
    thin = ApplyConvolution(
        kernel_kind="line",
        length=8.0,
        theta_deg=theta,
        sigma_perp=0.2,
        kernel_size=31,
        units="psf_pix",
    ).apply(psf)
    thick = ApplyConvolution(
        kernel_kind="line",
        length=8.0,
        theta_deg=theta,
        sigma_perp=1.0,
        kernel_size=31,
        units="psf_pix",
    ).apply(psf)

    cross_theta = theta + 90.0
    cross_thin = _projected_variance(thin.data, cross_theta)
    cross_thick = _projected_variance(thick.data, cross_theta)
    assert cross_thick > cross_thin


def test_apply_convolution_line_detector_pix_units_scale_to_psf_pixels():
    detector_units = ApplyConvolution(
        kernel_kind="line",
        length=2.0,
        theta_deg=20.0,
        sigma_perp=0.25,
        kernel_size=21,
        units="detector_pix",
        detector_to_psf_scale=3.0,
    )
    psf_units = ApplyConvolution(
        kernel_kind="line",
        length=6.0,
        theta_deg=20.0,
        sigma_perp=0.75,
        kernel_size=21,
        units="psf_pix",
    )

    assert jnp.allclose(
        detector_units.generate_kernel(),
        psf_units.generate_kernel(),
        atol=1e-6,
        rtol=1e-6,
    )


def test_apply_convolution_line_is_reasonable_approximation_to_shift_and_coadd_smear():
    # Approximate equivalence test: line-kernel convolution integrates motion
    # continuously, while this reference uses a small discrete shift-and-coadd.
    # We only require close agreement to modest tolerance.
    psf0 = _gaussian_image(n=81, sigma=2.0)
    psf0 = psf0 / jnp.sum(psf0)
    psf = PSF(data=psf0, pixel_scale=1.0)

    length = 6.0
    theta_deg = 30.0
    sigma_perp = 0.25
    conv_layer = ApplyConvolution(
        kernel_kind="line",
        length=length,
        theta_deg=theta_deg,
        sigma_perp=sigma_perp,
        kernel_size=31,
        units="psf_pix",
    )
    conv_out = conv_layer.apply(psf).data

    theta = jnp.deg2rad(theta_deg)
    ux = jnp.cos(theta)
    uy = jnp.sin(theta)
    samples = 25
    offsets = jnp.linspace(-0.5 * length, 0.5 * length, samples)
    coadd = jnp.zeros_like(psf0)
    for t in offsets:
        coadd = coadd + _shift_image_linear(psf0, dx=float(t * ux), dy=float(t * uy))
    coadd = coadd / jnp.sum(coadd)

    assert jnp.isclose(jnp.sum(conv_out), jnp.sum(coadd), atol=1e-6, rtol=1e-6)
    assert jnp.allclose(conv_out, coadd, atol=3e-2, rtol=6e-2)


@pytest.mark.parametrize("length", [0.0, -1.0])
def test_apply_convolution_line_invalid_length_raises(length):
    with pytest.raises(ValueError, match="length must be positive"):
        ApplyConvolution(
            kernel_kind="line",
            length=length,
            theta_deg=0.0,
            sigma_perp=0.5,
            kernel_size=9,
            units="psf_pix",
        )


@pytest.mark.parametrize("sigma_perp", [0.0, -0.1])
def test_apply_convolution_line_invalid_sigma_perp_raises(sigma_perp):
    with pytest.raises(ValueError, match="sigma_perp must be positive"):
        ApplyConvolution(
            kernel_kind="line",
            length=3.0,
            theta_deg=0.0,
            sigma_perp=sigma_perp,
            kernel_size=9,
            units="psf_pix",
        )


@pytest.mark.parametrize("kernel_size", [0, 2, -3])
def test_apply_convolution_invalid_kernel_size_raises(kernel_size):
    with pytest.raises(ValueError, match="positive odd integer"):
        ApplyConvolution(
            kernel_kind="gaussian",
            sigma_x=1.0,
            sigma_y=1.0,
            theta_deg=0.0,
            kernel_size=kernel_size,
            units="psf_pix",
        )


@pytest.mark.parametrize(
    ("sigma_x", "sigma_y"),
    [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -1.0)],
)
def test_apply_convolution_invalid_sigma_raises(sigma_x, sigma_y):
    with pytest.raises(ValueError, match="must be positive"):
        ApplyConvolution(
            kernel_kind="gaussian",
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            theta_deg=0.0,
            kernel_size=5,
            units="psf_pix",
        )


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


def test_apply_pixel_offsets_subpixel_linear_shift_matches_expected_ramp():
    width = 8
    height = 6
    y, x = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
    image = x.astype(float)
    psf = PSF(data=image, pixel_scale=1.0)

    dx = jnp.full_like(image, 0.5, dtype=float)
    dy = jnp.zeros_like(image, dtype=float)
    layer = ApplyPixelOffsets(dx_map=dx, dy_map=dy, interp_method="linear")
    out = layer.apply(psf)

    expected = jnp.clip(x + 0.5, 0.0, width - 1.0)
    assert jnp.allclose(out.data, expected, atol=1e-6, rtol=1e-6)


def test_apply_pixel_offsets_default_interp_is_cubic():
    image = _gaussian_image()
    layer = ApplyPixelOffsets(dx_map=jnp.zeros_like(image), dy_map=jnp.zeros_like(image))
    assert layer.interp_method == "cubic"


def test_apply_pixel_offsets_grad_stable_for_cubic_identity():
    image = _gaussian_image()
    layer = ApplyPixelOffsets(
        dx_map=jnp.zeros_like(image),
        dy_map=jnp.zeros_like(image),
        interp_method="cubic",
    )

    def loss(img):
        psf = PSF(data=img, pixel_scale=1.0)
        out = layer.apply(psf)
        return jnp.sum(out.data ** 2)

    grad = jax.grad(loss)(image)
    assert jnp.isfinite(grad).all()


def test_build_detector_layer_pixel_offsets_default_interp_is_cubic():
    layer_name, layer_obj = build_detector_layer(
        {"name": "pixel_offsets", "kind": "ApplyPixelOffsets"},
        target_shape=(5, 5),
        base_seed=None,
    )
    assert layer_name == "pixel_offsets"
    assert isinstance(layer_obj, ApplyPixelOffsets)
    assert layer_obj.interp_method == "cubic"


def test_build_detector_layer_apply_convolution_builds_gaussian_layer():
    layer_name, layer_obj = build_detector_layer(
        {
            "name": "diffusion",
            "kind": "ApplyConvolution",
            "kernel": {
                "kind": "gaussian",
                "sigma_x": 0.3,
                "sigma_y": 0.2,
                "theta_deg": 15.0,
                "kernel_size": 9,
                "units": "detector_pix",
            },
        },
        target_shape=(9, 9),
        base_seed=None,
        detector_to_psf_scale=3.0,
    )

    assert layer_name == "diffusion"
    assert isinstance(layer_obj, ApplyConvolution)
    assert layer_obj.kernel_kind == "gaussian"
    assert float(layer_obj.sigma_x) == 0.3
    assert float(layer_obj.sigma_y) == 0.2
    assert float(layer_obj.theta_deg) == 15.0
    assert int(layer_obj.kernel_size) == 9
    assert layer_obj.units == "detector_pix"
    assert float(layer_obj.detector_to_psf_scale) == 3.0


def test_build_detector_layer_apply_convolution_builds_box_layer():
    layer_name, layer_obj = build_detector_layer(
        {
            "name": "pixel_mtf",
            "kind": "ApplyConvolution",
            "kernel": {
                "kind": "box",
                "width_x": 1.0,
                "width_y": 0.7,
                "kernel_size": 9,
                "units": "detector_pix",
            },
        },
        target_shape=(9, 9),
        base_seed=None,
        detector_to_psf_scale=2.0,
    )

    assert layer_name == "pixel_mtf"
    assert isinstance(layer_obj, ApplyConvolution)
    assert layer_obj.kernel_kind == "box"
    assert float(layer_obj.width_x) == 1.0
    assert float(layer_obj.width_y) == 0.7
    assert int(layer_obj.kernel_size) == 9
    assert layer_obj.units == "detector_pix"
