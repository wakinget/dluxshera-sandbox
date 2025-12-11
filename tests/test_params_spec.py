import pytest

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


def _toy_spec() -> ParamSpec:
    return ParamSpec(
        (
            ParamField(key="a", group="g1", kind="primitive"),
            ParamField(key="b", group="g1", kind="primitive"),
            ParamField(key="primary.zernike_coeffs", group="primary", kind="primitive", shape=(3,)),
            ParamField(key="c", group="g2", kind="derived"),
        )
    )


def test_param_spec_without_basic_complement():
    spec = _toy_spec()
    excluded = spec.without(["b"])
    assert list(excluded.keys()) == ["a", "primary.zernike_coeffs", "c"]

    no_exclusions = spec.without([])
    assert list(no_exclusions.keys()) == list(spec.keys())


def test_param_spec_without_matches_subset_complement():
    spec = _toy_spec()
    all_keys = list(spec.keys())
    exclude_keys = {"b", "c"}
    keep_keys = [k for k in all_keys if k not in exclude_keys]

    assert list(spec.without(exclude_keys).keys()) == keep_keys
    assert list(spec.without(exclude_keys).keys()) == list(spec.subset(keep_keys).keys())


def test_param_spec_without_unknown_key_raises():
    spec = _toy_spec()
    with pytest.raises(KeyError):
        spec.without(["not_there"])


def test_param_spec_without_vector_field_removes_whole_field():
    spec = _toy_spec()
    excluded = spec.without(["primary.zernike_coeffs"])
    assert "primary.zernike_coeffs" not in excluded
    assert list(excluded.keys()) == ["a", "b", "c"]


def test_param_spec_without_all_keys_leaves_empty_spec():
    spec = _toy_spec()
    empty = spec.without(spec.keys())
    assert len(empty) == 0
