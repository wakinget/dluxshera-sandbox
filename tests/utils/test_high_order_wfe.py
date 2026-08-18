import numpy as np

from dluxshera.utils.high_order_wfe import (
    generate_high_order_wfe_map,
    one_over_f_noise_2d,
    realize_high_order_wfe_pair,
    white_noise_2d,
)


def test_noise_shapes_and_seed_reproducibility():
    a = one_over_f_noise_2d((32, 32), alpha=2.0, seed=1)
    b = one_over_f_noise_2d((32, 32), alpha=2.0, seed=1)
    c = one_over_f_noise_2d((32, 32), alpha=2.0, seed=2)
    assert a.shape == (32, 32)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)
    assert white_noise_2d((16, 16), seed=1).shape == (16, 16)


def test_rms_normalization_and_pair():
    mask = np.ones((32, 32), dtype=bool)
    arr, meta = generate_high_order_wfe_map((32, 32), kind="white", rms_nm=3.0, seed=1, remove_zernike_noll=[1,2,3], mask=mask)
    rms = float(np.sqrt(np.mean(arr[mask] ** 2)))
    assert abs(rms - 3.0) < 1e-6
    assert meta["units"] == "nm"

    r0 = realize_high_order_wfe_pair((32, 32), {"kind": "white", "rms_nm": 1.0, "seed": 1}, {"enabled": False})
    assert np.allclose(r0.truth_opd_nm, r0.inference_opd_nm)

    r1 = realize_high_order_wfe_pair((32, 32), {"kind": "white", "rms_nm": 1.0, "seed": 1}, {"enabled": True, "kind": "white", "rms_nm": 0.2, "seed": 3})
    assert not np.allclose(r1.truth_opd_nm, r1.inference_opd_nm)
