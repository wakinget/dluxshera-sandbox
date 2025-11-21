from dluxshera.params.spec import ParamField, ParamSpec


def test_param_spec_add_and_get():
    f = ParamField(
        key="imaging.psf_pixel_scale",
        group="imaging",
        kind="derived",
        units="arcsec / pixel",
        doc="On-sky PSF pixel scale."
    )

    spec = ParamSpec().add(f)

    assert "imaging.psf_pixel_scale" in spec
    retrieved = spec.get("imaging.psf_pixel_scale")
    assert retrieved.key == f.key
    assert retrieved.group == "imaging"
