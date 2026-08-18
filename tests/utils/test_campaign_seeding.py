from __future__ import annotations

import pytest

from dluxshera.utils.seeding import derive_campaign_subblock_seeds


def test_derive_campaign_subblock_seeds_is_deterministic() -> None:
    a = derive_campaign_subblock_seeds(
        base_seed=42,
        seed_policy="different_jitter_different_noise",
        campaign_token="run",
        case_token="case_a",
        subblock_index=0,
    )
    b = derive_campaign_subblock_seeds(
        base_seed=42,
        seed_policy="different_jitter_different_noise",
        campaign_token="run",
        case_token="case_a",
        subblock_index=0,
    )
    assert a == b


def test_same_jitter_different_noise_policy() -> None:
    a0 = derive_campaign_subblock_seeds(
        base_seed=42,
        seed_policy="same_jitter_different_noise",
        campaign_token="run",
        case_token="case_a",
        subblock_index=0,
    )
    a1 = derive_campaign_subblock_seeds(
        base_seed=42,
        seed_policy="same_jitter_different_noise",
        campaign_token="run",
        case_token="case_a",
        subblock_index=1,
    )
    assert a0.trace_seed == a1.trace_seed
    assert a0.noise_seed != a1.noise_seed


def test_different_jitter_same_noise_policy() -> None:
    a0 = derive_campaign_subblock_seeds(
        base_seed=42,
        seed_policy="different_jitter_same_noise",
        campaign_token="run",
        case_token="case_a",
        subblock_index=0,
    )
    a1 = derive_campaign_subblock_seeds(
        base_seed=42,
        seed_policy="different_jitter_same_noise",
        campaign_token="run",
        case_token="case_a",
        subblock_index=1,
    )
    assert a0.trace_seed != a1.trace_seed
    assert a0.noise_seed == a1.noise_seed


def test_invalid_policy_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported seed_policy"):
        derive_campaign_subblock_seeds(
            base_seed=42,
            seed_policy="bad_policy",
            campaign_token="run",
            case_token="case",
            subblock_index=0,
        )
