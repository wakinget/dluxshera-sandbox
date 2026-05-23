"""Shared campaign seed derivation helpers."""

from __future__ import annotations

from dataclasses import dataclass

from dluxshera.utils.noise import make_subseed

SUPPORTED_CAMPAIGN_SEED_POLICIES = (
    "different_jitter_different_noise",
    "same_jitter_different_noise",
    "different_jitter_same_noise",
)


@dataclass(frozen=True)
class CampaignSeeds:
    trace_seed: int
    noise_seed: int
    subblock_seed: int
    policy: str
    base_seed: int
    token: str
    case_token: str
    campaign_token: str


def derive_campaign_subblock_seeds(
    *,
    base_seed: int,
    seed_policy: str,
    campaign_token: str,
    case_token: str,
    subblock_index: int | None = None,
    trial_index: int | None = None,
) -> CampaignSeeds:
    if seed_policy not in SUPPORTED_CAMPAIGN_SEED_POLICIES:
        raise ValueError(f"Unsupported seed_policy: {seed_policy}")
    index = subblock_index if subblock_index is not None else trial_index
    if index is None:
        raise ValueError("Either subblock_index or trial_index must be provided.")
    token = f"{campaign_token}.{case_token}.subblock_{int(index):03d}"
    subblock_seed = make_subseed(int(base_seed), token)
    if seed_policy == "same_jitter_different_noise":
        trace_seed = make_subseed(int(base_seed), f"{campaign_token}.{case_token}.shared_trace")
        noise_seed = make_subseed(int(base_seed), f"{token}.noise")
    elif seed_policy == "different_jitter_same_noise":
        trace_seed = make_subseed(int(base_seed), f"{token}.trace")
        noise_seed = make_subseed(int(base_seed), f"{campaign_token}.{case_token}.shared_noise")
    else:
        trace_seed = make_subseed(int(base_seed), f"{token}.trace")
        noise_seed = make_subseed(int(base_seed), f"{token}.noise")
    return CampaignSeeds(
        trace_seed=int(trace_seed),
        noise_seed=int(noise_seed),
        subblock_seed=int(subblock_seed),
        policy=str(seed_policy),
        base_seed=int(base_seed),
        token=token,
        case_token=str(case_token),
        campaign_token=str(campaign_token),
    )
