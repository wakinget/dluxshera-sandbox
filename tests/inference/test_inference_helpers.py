import pytest

from dluxshera.params.spec import ParamField, ParamSpec, make_inference_subspec
from dluxshera.params.store import ParameterStore, validate_inference_base_store
from dluxshera.optics.config import SheraThreePlaneConfig, SheraTwoPlaneConfig


def _base_spec():
    return ParamSpec(
        [
            ParamField("a", group="g", kind="primitive"),
            ParamField("primary.zernike_coeffs_nm", group="primary", kind="primitive", shape=(2,)),
            ParamField("secondary.zernike_coeffs_nm", group="secondary", kind="primitive", shape=(1,)),
        ]
    )


def test_make_inference_subspec_preserves_order():
    spec = _base_spec()
    infer_keys = ["secondary.zernike_coeffs_nm", "a"]

    subspec = make_inference_subspec(base_spec=spec, infer_keys=infer_keys)

    assert list(subspec.keys()) == infer_keys


def test_make_inference_subspec_raises_when_basis_missing():
    spec = _base_spec()
    cfg = SheraTwoPlaneConfig(primary_noll_indices=())

    with pytest.raises(ValueError):
        make_inference_subspec(
            base_spec=spec, infer_keys=["primary.zernike_coeffs_nm"], cfg=cfg
        )


def test_make_inference_subspec_respects_include_secondary_false():
    spec = _base_spec()
    cfg = SheraThreePlaneConfig(primary_noll_indices=(4,), secondary_noll_indices=(4,))

    with pytest.raises(ValueError):
        make_inference_subspec(
            base_spec=spec,
            infer_keys=["secondary.zernike_coeffs_nm"],
            cfg=cfg,
            include_secondary=False,
        )


def test_validate_inference_base_store_passes_when_keys_and_shapes_match():
    spec = ParamSpec(
        [
            ParamField("x", group="g", kind="primitive", shape=None),
            ParamField("y", group="g", kind="primitive", shape=(2,)),
        ]
    )
    store = ParameterStore.from_dict({"x": 1.0, "y": [0.0, 1.0]})

    validate_inference_base_store(store, spec)


def test_validate_inference_base_store_raises_on_missing_key():
    spec = ParamSpec([ParamField("x", group="g", kind="primitive")])
    store = ParameterStore.from_dict({})

    with pytest.raises(ValueError) as excinfo:
        validate_inference_base_store(store, spec)

    assert "missing keys" in str(excinfo.value)


def test_validate_inference_base_store_shape_mismatch():
    spec = ParamSpec([ParamField("y", group="g", kind="primitive", shape=(3,))])
    store = ParameterStore.from_dict({"y": [0.0, 1.0]})

    with pytest.raises(ValueError) as excinfo:
        validate_inference_base_store(store, spec)

    assert "shape mismatches" in str(excinfo.value)

