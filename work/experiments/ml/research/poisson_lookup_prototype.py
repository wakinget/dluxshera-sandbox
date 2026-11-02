#!/usr/bin/env python3
"""RESEARCH PROTOTYPE: precomputed Poisson lookup tables.

**Not production code.** Nothing here is imported by ``src/dluxshera``, the S12
DataLoader, or any training path, and it must stay that way.  It exists to
answer one question quantitatively: can a table of precomputed Poisson draws,
indexed cheaply, replace a live Poisson sampler for S12-scale workloads without
unacceptable statistical cost?

The production pathway remains the exact batched JAX implementation in
``dluxshera.ml.noise.apply_pair_noise_batch``.  A lookup table cannot reproduce
the prescribed JAX realization bitwise; see ``independence`` and ``fidelity``
below for what it can and cannot claim.

Table model
-----------
``T[k, u, v]`` holds an independent Poisson draw at expected-count level ``k``.
``(u, v)`` are table coordinates, deliberately decoupled from detector pixel
coordinates: the table's spatial extent is a sample-pool size, not an image
size.  The count-level axis is placed first so that one level's pool is
contiguous, which is what makes the gather cache-friendly.

Indexing strategies
-------------------
``per_pixel``
    Draw an independent ``(u, v)`` for every output pixel.  Maximum sampling
    freedom, but it needs two random integers per pixel and produces a fully
    irregular gather.
``offset``
    Draw one ``(dx, dy)`` per image (optionally per count level) and index
    ``((x + dx) % W, (y + dy) % H)``.  A handful of random values per image and
    a far more regular access pattern, at the cost of structured -- not
    independent -- sampling positions within each image.

    **Shared-offset caveat.**  With ``offset_per_level=True`` the translations
    are drawn once per *count level* and reused for every image in the call, so
    two identical expected images in one batch receive the **same** draw, not
    two independent ones.  ``correlation_report``'s duplicate-input probe
    measures this directly.  The default ``offset_per_level=False`` draws one
    translation per image and does not have this defect, but still reuses whole
    translated regions across images.

Fractional means
----------------
Expected counts are generally non-integer.  ``nearest`` quantizes to the closest
stored level and is biased by the quantization error.  ``mixture`` randomizes
between the two bracketing integer levels with probability set by the
fractional part, which is unbiased in the mean but *not* Poisson-distributed:
a mixture of Poisson(k) and Poisson(k+1) is overdispersed relative to
Poisson(k+f).  ``exact_poisson`` is the reference sampler, included so every
measurement has a ground truth in the same harness.

Exactness, and its limits
-------------------------
``DyadicPoissonTable`` is the one construction here whose output is
*distributionally* Poisson, and that claim is conditional, not unqualified.  It
holds for one draw at a lambda lying on the ``delta`` lattice, assuming the
summed pool entries are independent.  A finite table cannot supply unlimited
independent draws: entries are revisited across a campaign (see
``reuse_report``), so the independence assumption underpinning the exactness
degrades with reuse.  Lambda off the lattice is quantized, and lambda above the
representable maximum is clipped.  Read "exact" throughout this module as "exact
under the stated idealization", never as "equivalent to unlimited independent
Poisson draws".

Run ``--help`` for the measurement entry points.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

#: Strategies compared throughout this module.
INDEX_STRATEGIES = ("per_pixel", "offset")

#: Fractional-mean treatments compared throughout this module.
LEVEL_POLICIES = ("nearest", "mixture")

#: Default S12-shaped image geometry.
IMAGE_SHAPE = (160, 160)

#: Refuse to allocate more than this without an explicit opt-in.  A
#: 1024x1024x1024 uint16 table is 2 GiB; defaults must never reach for that.
DEFAULT_MEMORY_BUDGET_MIB = 512.0


# ---------------------------------------------------------------------------
# Table construction and memory accounting
# ---------------------------------------------------------------------------
def table_bytes(*, levels: int, width: int, height: int, dtype: str) -> int:
    """Return the exact byte size of a ``T[k, u, v]`` table."""
    return int(levels) * int(width) * int(height) * int(np.dtype(dtype).itemsize)


def dtype_is_safe_for_levels(*, max_level: float, dtype: str, sigma: float = 8.0) -> dict:
    """Check that Poisson draws at ``max_level`` fit the chosen integer dtype.

    Poisson(lambda) has standard deviation ``sqrt(lambda)``, with no hard upper
    bound, so a dtype is only ever safe to a stated number of sigma.  At
    ``lambda = 60000`` a uint16 table is already within ~2.5 sigma of overflow;
    silently wrapping would corrupt the brightest pixels, which are exactly the
    ones carrying the most signal.
    """
    info = np.iinfo(np.dtype(dtype)) if np.dtype(dtype).kind in "iu" else None
    if info is None:
        return {"dtype": dtype, "integer": False, "safe": True,
                "note": "floating dtype; no integer overflow, but 4x the memory of uint16"}
    headroom = float(max_level) + sigma * float(np.sqrt(max(max_level, 0.0)))
    return {
        "dtype": dtype,
        "integer": True,
        "dtype_max": int(info.max),
        "max_level": float(max_level),
        f"level_plus_{sigma:g}_sigma": headroom,
        "safe": bool(headroom <= info.max),
        "note": "Poisson is unbounded above; safety is only to the stated sigma",
    }


@dataclass
class PoissonTable:
    """A precomputed Poisson sample pool with a declared count-level grid.

    Parameters
    ----------
    levels:
        Expected-count values stored, ascending.  Integer-valued for the
        ``nearest``/``mixture`` policies.
    width, height:
        Table spatial extent.  Independent of image size.
    dtype:
        Storage dtype.  ``uint16`` halves memory relative to ``float32``.
    seed:
        Table-generation seed.  Part of the reproducibility schedule: the same
        seed must yield the same table, or nothing downstream is reproducible.
    """

    levels: np.ndarray
    width: int
    height: int
    dtype: str = "uint16"
    seed: int = 0
    values: np.ndarray = field(init=False)
    build_seconds: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.levels = np.asarray(self.levels, dtype=np.float64)
        nbytes = table_bytes(levels=len(self.levels), width=self.width,
                             height=self.height, dtype=self.dtype)
        started = time.perf_counter()
        rng = np.random.default_rng(int(self.seed))
        draws = rng.poisson(
            self.levels[:, None, None],
            size=(len(self.levels), int(self.height), int(self.width)),
        )
        if np.dtype(self.dtype).kind in "iu":
            limit = np.iinfo(np.dtype(self.dtype)).max
            if int(draws.max()) > limit:
                raise ValueError(
                    f"Generated Poisson draws reach {int(draws.max())}, which does "
                    f"not fit {self.dtype} (max {limit}). Choose a wider dtype."
                )
        self.values = np.ascontiguousarray(draws.astype(self.dtype))
        self.build_seconds = time.perf_counter() - started
        self._nbytes = nbytes

    @property
    def nbytes(self) -> int:
        return int(self.values.nbytes)

    def describe(self) -> dict:
        return {
            "levels": len(self.levels),
            "level_min": float(self.levels.min()),
            "level_max": float(self.levels.max()),
            "spatial": [int(self.height), int(self.width)],
            "draws_per_level": int(self.width) * int(self.height),
            "dtype": self.dtype,
            "mib": self.nbytes / 1024**2,
            "build_seconds": self.build_seconds,
            "seed": int(self.seed),
            "dtype_safety": dtype_is_safe_for_levels(
                max_level=float(self.levels.max()), dtype=self.dtype),
        }


def integer_levels(max_level: int) -> np.ndarray:
    """Unit-spaced integer levels ``0..max_level``."""
    return np.arange(0, int(max_level) + 1, dtype=np.float64)


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------
def _level_index(expected: np.ndarray, table: PoissonTable) -> np.ndarray:
    """Nearest stored level index for each expected count, clipped to the grid.

    Clipping is the dangerous part and is why :func:`level_coverage_report`
    exists.  A pixel whose expected count exceeds the top stored level is
    silently given a draw from that top level instead, which is not an
    approximation of its distribution at all -- it is the wrong distribution.
    """
    return np.clip(
        np.rint(expected).astype(np.int64), 0, len(table.levels) - 1
    )


def level_coverage_report(expected: np.ndarray, table: PoissonTable) -> dict:
    """Report how much of ``expected`` falls outside the stored level grid.

    This is the first gate any lookup configuration must pass.  Real SHERA
    images span several decades of counts, so a table sized for the typical
    pixel silently truncates the bright core, where most of the signal is.
    """
    arr = np.asarray(expected, dtype=np.float64)
    top = float(table.levels.max())
    over = arr > top
    needed = int(np.ceil(arr.max())) + 1
    return {
        "stored_levels": int(len(table.levels)),
        "top_stored_level": top,
        "image_max_expected_count": float(arr.max()),
        "fraction_pixels_above_top_level": float(np.mean(over)),
        "pixels_above_top_level": int(over.sum()),
        "clipped_counts_are_wrong_distribution": bool(over.any()),
        "levels_required_for_full_coverage": needed,
        "uint16_mib_for_full_coverage_at_this_spatial_size": table_bytes(
            levels=needed, width=table.width, height=table.height, dtype="uint16"
        ) / 1024**2,
        "note": (
            "Pixels above the top stored level receive draws from that level, "
            "not from their own. Full coverage of a several-decade count range "
            "is what drives the table into the multi-GiB regime."
        ),
    }


def _mixture_level_index(
    expected: np.ndarray, table: PoissonTable, rng: np.random.Generator
) -> np.ndarray:
    """Randomized bracketing level index: floor(lambda) or ceil(lambda).

    Picking ``k+1`` with probability equal to the fractional part makes the
    *level* mean correct.  It does not make the result Poisson: the realized
    distribution is a two-component Poisson mixture, whose variance exceeds
    Poisson(lambda) by ``f(1-f)``.
    """
    clipped = np.clip(expected, 0.0, float(len(table.levels) - 1))
    low = np.floor(clipped).astype(np.int64)
    frac = clipped - low
    bump = (rng.random(size=clipped.shape) < frac).astype(np.int64)
    return np.clip(low + bump, 0, len(table.levels) - 1)


def sample_lookup(
    expected: np.ndarray,
    table: PoissonTable,
    *,
    strategy: str = "per_pixel",
    level_policy: str = "nearest",
    rng: np.random.Generator,
    offset_per_level: bool = False,
    return_coords: bool = False,
):
    """Draw a noisy image by indexing ``table`` instead of sampling Poisson.

    Parameters
    ----------
    expected:
        Expected counts, shape ``(..., H, W)``.  Treated as electrons under the
        accepted unity DN/electron conversion.
    strategy:
        ``"per_pixel"`` or ``"offset"``; see the module docstring.
    level_policy:
        ``"nearest"`` or ``"mixture"``; see the module docstring.
    offset_per_level:
        ``"offset"`` only.  When ``True``, each count level gets its own
        translation, which decorrelates levels at the cost of more random
        values and a less regular gather.
    return_coords:
        Also return the realized ``(u, v)`` table coordinates, for collision
        and reuse accounting.

    Returns
    -------
    numpy.ndarray or tuple
        Sampled counts as ``float32``, matching ``expected.shape``.
    """
    if strategy not in INDEX_STRATEGIES:
        raise ValueError(f"strategy must be one of {INDEX_STRATEGIES}, got {strategy!r}.")
    if level_policy not in LEVEL_POLICIES:
        raise ValueError(f"level_policy must be one of {LEVEL_POLICIES}, got {level_policy!r}.")
    arr = np.asarray(expected, dtype=np.float64)
    k = (
        _level_index(arr, table)
        if level_policy == "nearest"
        else _mixture_level_index(arr, table, rng)
    )

    if strategy == "per_pixel":
        u = rng.integers(0, table.height, size=arr.shape)
        v = rng.integers(0, table.width, size=arr.shape)
    else:
        rows, cols = arr.shape[-2:]
        y = np.arange(rows)[:, None]
        x = np.arange(cols)[None, :]
        if offset_per_level:
            # One translation per stored level, selected by the level index, so
            # two levels never share a translation within an image.
            dy = rng.integers(0, table.height, size=len(table.levels))
            dx = rng.integers(0, table.width, size=len(table.levels))
            u = (y + dy[k]) % table.height
            v = (x + dx[k]) % table.width
        else:
            # One translation per image. A bare 2D input is a single image, so
            # the leading axes may be empty; reshape to broadcast either way.
            lead = arr.shape[:-2]
            dy = rng.integers(0, table.height, size=lead or ())
            dx = rng.integers(0, table.width, size=lead or ())
            shape = tuple(lead) + (1, 1)
            u = np.broadcast_to(
                (y + np.reshape(dy, shape)) % table.height, arr.shape)
            v = np.broadcast_to(
                (x + np.reshape(dx, shape)) % table.width, arr.shape)

    values = table.values[k, u, v].astype(np.float32)
    if return_coords:
        return values, k, np.asarray(u), np.asarray(v)
    return values


def sample_exact_numpy(
    expected: np.ndarray, *, rng: np.random.Generator
) -> np.ndarray:
    """Vectorized NumPy Poisson reference, used as the fidelity ground truth."""
    return rng.poisson(np.asarray(expected, dtype=np.float64)).astype(np.float32)


# ---------------------------------------------------------------------------
# The one exact construction: dyadic superposition
# ---------------------------------------------------------------------------
@dataclass
class DyadicPoissonTable:
    """Tables at dyadic rates, composable into an exactly Poisson result.

    Poisson is infinitely divisible and closed under addition of independent
    variates: ``Poisson(a) + Poisson(b) == Poisson(a + b)`` in distribution.  So
    if the table stores one pool per rate ``2**j * delta`` and the requested
    ``lambda`` is written in that dyadic basis, summing *independent* draws from
    the selected pools yields a draw that is exactly ``Poisson(lambda_q)``,
    where ``lambda_q`` is ``lambda`` rounded to the resolution ``delta``.

    This is categorically different from ``nearest`` and ``mixture``: the
    distribution is correct, not merely close, and the only error is the
    representation of ``lambda`` itself at resolution ``delta``.  The price is
    one gather per set bit -- roughly half the stored rates per pixel -- which
    is what :func:`exact_construction_report` measures.

    Parameters
    ----------
    delta:
        Finest rate, and the quantization resolution of ``lambda``.
    rates:
        Number of dyadic rates stored, covering ``delta`` to ``delta * 2**(n-1)``.
    width, height:
        Pool extent per rate.
    """

    delta: float
    rates: int
    width: int
    height: int
    dtype: str = "uint16"
    seed: int = 0
    values: np.ndarray = field(init=False)
    build_seconds: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.rate_values = self.delta * 2.0 ** np.arange(int(self.rates))
        started = time.perf_counter()
        rng = np.random.default_rng(int(self.seed))
        draws = rng.poisson(
            self.rate_values[:, None, None],
            size=(int(self.rates), int(self.height), int(self.width)),
        )
        self.values = np.ascontiguousarray(draws.astype(self.dtype))
        self.build_seconds = time.perf_counter() - started

    @property
    def max_lambda(self) -> float:
        return float(self.rate_values.sum())

    def sample(
        self, expected: np.ndarray, *, rng: np.random.Generator, strategy: str = "offset"
    ) -> tuple[np.ndarray, int]:
        """Sum independent pool draws for each set dyadic bit.

        Returns the sampled counts and the number of gathers performed, because
        the gather count is the cost that decides this approach.
        """
        arr = np.asarray(expected, dtype=np.float64)
        # Quantize lambda to the dyadic lattice, then read its bits.
        units = np.minimum(
            np.rint(arr / self.delta).astype(np.int64),
            int(round(self.max_lambda / self.delta)),
        )
        total = np.zeros(arr.shape, dtype=np.float32)
        gathers = 0
        rows, cols = arr.shape[-2:]
        y = np.arange(rows)[:, None]
        x = np.arange(cols)[None, :]
        for bit in range(int(self.rates)):
            selected = (units >> bit) & 1
            if not selected.any():
                continue
            # Each rate gets its own independent translation, so the summands
            # are independent draws rather than correlated reads of one pool.
            if strategy == "offset":
                dy, dx = rng.integers(0, self.height), rng.integers(0, self.width)
                u = np.broadcast_to((y + dy) % self.height, arr.shape)
                v = np.broadcast_to((x + dx) % self.width, arr.shape)
            else:
                u = rng.integers(0, self.height, size=arr.shape)
                v = rng.integers(0, self.width, size=arr.shape)
            total += selected * self.values[bit, u, v]
            gathers += 1
        return total, gathers


def exact_construction_report(
    images: np.ndarray,
    *,
    delta: float,
    width: int,
    height: int,
    repeats: int,
    seed: int,
    lambdas: tuple[float, ...] = (0.1, 0.37, 1.0, 5.0, 12.37, 20.0, 300.0),
    draws: int = 200_000,
) -> dict:
    """Measure the dyadic construction's fidelity and its cost.

    The point of this report is the trade: fidelity becomes a non-issue, and the
    gather count per pixel becomes the binding cost.
    """
    from scipy.stats import chisquare, poisson

    needed = int(np.ceil(np.log2(max(float(images.max()), 1.0) / delta))) + 1
    table = DyadicPoissonTable(delta=delta, rates=needed, width=width,
                               height=height, seed=seed)
    rng = np.random.default_rng(seed)

    fidelity = []
    for lam in lambdas:
        side = int(np.ceil(np.sqrt(draws)))
        got, _ = table.sample(np.full((side, side), float(lam)), rng=rng)
        got = got.ravel()[:draws]
        high = int(poisson.ppf(1.0 - 1e-9, max(lam, 1e-3))) + 2
        edges = np.arange(0, high + 1)
        observed = np.bincount(np.clip(got.astype(np.int64), 0, high), minlength=high + 1)
        pmf = _poisson_pmf(edges, max(float(lam), 1e-12))
        pmf = pmf / pmf.sum()
        exp_counts = pmf * draws
        keep = exp_counts >= 5.0
        if keep.sum() >= 2:
            obs = np.append(observed[keep], observed[~keep].sum())
            exp = np.append(exp_counts[keep], exp_counts[~keep].sum())
            exp = exp * (obs.sum() / exp.sum())
            chi2, pvalue = chisquare(obs, exp)
        else:
            chi2, pvalue = float("nan"), float("nan")
        fidelity.append({
            "lambda": float(lam),
            "mean": float(got.mean()),
            "variance": float(got.var()),
            "mean_relative_error": float((got.mean() - lam) / lam) if lam else None,
            "variance_relative_error": float((got.var() - lam) / lam) if lam else None,
            "p_zero": float(np.mean(got == 0)),
            "p_zero_theory": float(np.exp(-lam)),
            "total_variation_distance": 0.5 * float(np.abs(observed / draws - pmf).sum()),
            "chi2": float(chi2),
            "chi2_pvalue": float(pvalue),
        })

    counts = []
    timing = _time(lambda: counts.append(table.sample(images, rng=rng)[1]),
                   repeats=repeats)
    batch = images.shape[0]
    return {
        "construction": "dyadic superposition of independent Poisson pools",
        "exactness": (
            "distributionally exact for lambda on the delta lattice; Poisson is "
            "closed under addition of independent variates"
        ),
        "delta": float(delta),
        "rates_stored": needed,
        "max_representable_lambda": table.max_lambda,
        "pool_spatial": [int(height), int(width)],
        "table_mib": table.values.nbytes / 1024**2,
        "build_seconds": table.build_seconds,
        "gathers_per_call": max(counts) if counts else None,
        "mean_gathers_per_call": float(np.mean(counts)) if counts else None,
        "warm_seconds": timing["warm_seconds"],
        "seconds_per_32_image_batch": timing["warm_seconds"] * (32.0 / batch),
        "images_per_second": batch / timing["warm_seconds"],
        "fidelity": fidelity,
        "note": (
            "Cost scales with the number of set dyadic bits, so a fine delta "
            "and a wide count range both make it slower. Compare "
            "seconds_per_32_image_batch against numpy_poisson, not against the "
            "single-gather approximate variants."
        ),
    }


# ---------------------------------------------------------------------------
# Statistical quality
# ---------------------------------------------------------------------------
def _poisson_pmf(k: np.ndarray, lam: float) -> np.ndarray:
    from scipy.stats import poisson
    return poisson.pmf(k, lam)


def distribution_report(
    lam: float,
    table: PoissonTable,
    *,
    strategy: str,
    level_policy: str,
    draws: int,
    seed: int,
) -> dict:
    """Compare lookup draws at a single expected count against Poisson(lam)."""
    from scipy.stats import chisquare, poisson

    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(draws)))
    expected = np.full((side, side), float(lam))
    got = sample_lookup(expected, table, strategy=strategy,
                        level_policy=level_policy, rng=rng).ravel()[:draws]
    ref = sample_exact_numpy(expected, rng=np.random.default_rng(seed + 1)).ravel()[:draws]

    high = int(poisson.ppf(1.0 - 1e-9, max(lam, 1e-3))) + 2
    edges = np.arange(0, high + 1)
    observed = np.bincount(np.clip(got.astype(np.int64), 0, high), minlength=high + 1)
    pmf = _poisson_pmf(edges, max(float(lam), 1e-12))
    pmf = pmf / pmf.sum()
    expected_counts = pmf * draws
    # Pool bins with <5 expected counts; a chi-square on sparse tails is
    # dominated by rare-event noise rather than by distributional error.
    keep = expected_counts >= 5.0
    if keep.sum() >= 2:
        pooled_obs = np.append(observed[keep], observed[~keep].sum())
        pooled_exp = np.append(expected_counts[keep], expected_counts[~keep].sum())
        pooled_exp = pooled_exp * (pooled_obs.sum() / pooled_exp.sum())
        chi2, pvalue = chisquare(pooled_obs, pooled_exp)
        bins_used = int(keep.sum()) + 1
    else:
        chi2, pvalue, bins_used = float("nan"), float("nan"), 0

    tv = 0.5 * float(np.abs(observed / draws - pmf).sum())
    return {
        "lambda": float(lam),
        "strategy": strategy,
        "level_policy": level_policy,
        "draws": int(draws),
        "lookup": {
            "mean": float(got.mean()),
            "variance": float(got.var()),
            "fano": float(got.var() / got.mean()) if got.mean() > 0 else None,
            "p_zero": float(np.mean(got == 0)),
            "p_tail_above_2sigma": float(
                np.mean(got > lam + 2.0 * np.sqrt(max(lam, 1e-12)))),
        },
        "exact_numpy": {
            "mean": float(ref.mean()),
            "variance": float(ref.var()),
            "fano": float(ref.var() / ref.mean()) if ref.mean() > 0 else None,
            "p_zero": float(np.mean(ref == 0)),
            "p_tail_above_2sigma": float(
                np.mean(ref > lam + 2.0 * np.sqrt(max(lam, 1e-12)))),
        },
        "theory": {"mean": float(lam), "variance": float(lam),
                   "p_zero": float(np.exp(-lam))},
        "mean_relative_error": float((got.mean() - lam) / lam) if lam > 0 else None,
        "variance_relative_error": float((got.var() - lam) / lam) if lam > 0 else None,
        "total_variation_distance": tv,
        "chi2": float(chi2),
        "chi2_pvalue": float(pvalue),
        "chi2_bins_pooled_to": bins_used,
        "chi2_note": "bins with <5 expected counts pooled into one tail bin",
    }


def correlation_report(
    lam: float,
    table: PoissonTable,
    *,
    strategy: str,
    level_policy: str,
    images: int,
    shape: tuple[int, int],
    seed: int,
) -> dict:
    """Measure spatial and cross-realization correlation of lookup draws."""
    rng = np.random.default_rng(seed)
    expected = np.full((images, *shape), float(lam))
    got = sample_lookup(expected, table, strategy=strategy,
                        level_policy=level_policy, rng=rng)
    ref = sample_exact_numpy(expected, rng=np.random.default_rng(seed + 7))

    def lag_corr(stack: np.ndarray, dy: int, dx: int) -> float:
        a = stack[:, : stack.shape[1] - dy or None, : stack.shape[2] - dx or None]
        b = stack[:, dy:, dx:]
        a, b = a.ravel(), b.ravel()
        if a.std() == 0 or b.std() == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    # Two independently requested realizations of the same expected image: a
    # lookup scheme that reuses table entries will show correlation here.
    second = sample_lookup(expected, table, strategy=strategy,
                           level_policy=level_policy,
                           rng=np.random.default_rng(seed + 101))
    pairwise = [
        float(np.corrcoef(got[i].ravel(), second[i].ravel())[0, 1])
        for i in range(min(images, 32))
        if got[i].std() > 0 and second[i].std() > 0
    ]
    # Duplicate-input probe.  Correlation across *different* expected images
    # cannot detect a shared index: the expected values differ, so the outputs
    # differ regardless.  Feeding the batch identical expected images isolates
    # the indexing.  Any identical output pair means the requested
    # "independent" realizations are the same draw, not merely correlated.
    duplicate = np.full((4, *shape), float(lam))
    dup_draw = sample_lookup(duplicate, table, strategy=strategy,
                             level_policy=level_policy,
                             rng=np.random.default_rng(seed + 211))
    identical = [
        [i, j]
        for i in range(len(dup_draw))
        for j in range(i + 1, len(dup_draw))
        if np.array_equal(dup_draw[i], dup_draw[j])
    ]
    return {
        "lambda": float(lam),
        "strategy": strategy,
        "level_policy": level_policy,
        "images": int(images),
        "duplicate_input_independence": {
            "images_in_probe": int(len(dup_draw)),
            "identical_output_pairs": identical,
            "independent": not identical,
            "note": (
                "Identical expected images in one batch must still yield "
                "distinct draws. A non-empty list means the strategy shares one "
                "index across the batch, so requested independent realizations "
                "are literally the same sample."
            ),
        },
        "spatial_lag_correlation": {
            f"dy{dy}_dx{dx}": lag_corr(got, dy, dx)
            for dy, dx in ((0, 1), (1, 0), (1, 1), (0, 8), (8, 0))
        },
        "spatial_lag_correlation_exact_numpy": {
            f"dy{dy}_dx{dx}": lag_corr(ref, dy, dx)
            for dy, dx in ((0, 1), (1, 0), (1, 1), (0, 8), (8, 0))
        },
        "cross_realization_correlation": {
            "mean": float(np.mean(pairwise)) if pairwise else None,
            "max_abs": float(np.max(np.abs(pairwise))) if pairwise else None,
            "images_compared": len(pairwise),
        },
        "note": (
            "Exact-NumPy lag correlations give the finite-sample noise floor; "
            "lookup values must be compared against that, not against zero."
        ),
    }


def reuse_report(
    lam: float,
    table: PoissonTable,
    *,
    strategy: str,
    images: int,
    shape: tuple[int, int],
    seed: int,
) -> dict:
    """Quantify table-entry reuse within an image and across a campaign.

    The campaign figure is the one that matters.  A 1,048,576-entry slice is
    ample for approximating a one-dimensional distribution, and still tiny
    against S12's draw count: 600 epochs x 16,384 pairs x 25,600 pixels is
    roughly 2.5e11 draws, so every entry at a common count level is revisited
    on the order of 10^5 times.
    """
    rng = np.random.default_rng(seed)
    expected = np.full((images, *shape), float(lam))
    _, k, u, v = sample_lookup(expected, table, strategy=strategy,
                               level_policy="nearest", rng=rng, return_coords=True)
    pixels = int(np.prod(shape))
    flat = (np.asarray(k).ravel().astype(np.int64) * table.height * table.width
            + np.asarray(u).ravel().astype(np.int64) * table.width
            + np.asarray(v).ravel().astype(np.int64))
    per_image = []
    for i in range(min(images, 16)):
        chunk = flat[i * pixels : (i + 1) * pixels]
        per_image.append(1.0 - np.unique(chunk).size / chunk.size)
    slice_entries = table.height * table.width

    s12_draws = 600 * 16384 * pixels
    return {
        "lambda": float(lam),
        "strategy": strategy,
        "image_pixels": pixels,
        "table_entries_per_level": int(slice_entries),
        "pixels_per_slice_ratio": pixels / slice_entries,
        "within_image_reuse_fraction": {
            "mean": float(np.mean(per_image)),
            "max": float(np.max(per_image)),
            "images_measured": len(per_image),
        },
        "distinct_entries_touched_overall": int(np.unique(flat).size),
        "distinct_entry_fraction_of_level_pool": float(
            np.unique(flat).size / slice_entries),
        "s12_campaign_projection": {
            "draws_600_epochs_16384_pairs": int(s12_draws),
            "expected_reuse_per_entry_if_single_level": s12_draws / slice_entries,
            "note": (
                "A table entry at a frequently occurring count level is revisited "
                "on the order of 1e5 times across one S12 run; draws are therefore "
                "not independent at campaign scale regardless of index randomization."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
def _time(fn: Callable[[], object], *, repeats: int) -> dict:
    first_started = time.perf_counter()
    fn()
    first = time.perf_counter() - first_started
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    warm = (time.perf_counter() - started) / repeats
    return {"first_call_seconds": first, "warm_seconds": warm}


def performance_report(
    images: np.ndarray,
    table: PoissonTable,
    *,
    repeats: int,
    seed: int,
    include_jax: bool = True,
) -> dict:
    """Benchmark every candidate on one batch of expected-count images."""
    import resource

    batch = images.shape[0]
    pixels = int(np.prod(images.shape))
    rng = np.random.default_rng(seed)
    rows: dict[str, dict] = {}

    rows["numpy_poisson"] = _time(
        lambda: sample_exact_numpy(images, rng=rng), repeats=repeats)

    for strategy in INDEX_STRATEGIES:
        for policy in LEVEL_POLICIES:
            rows[f"lookup_{strategy}_{policy}"] = _time(
                lambda s=strategy, p=policy: sample_lookup(
                    images, table, strategy=s, level_policy=p, rng=rng),
                repeats=repeats)
    rows["lookup_offset_per_level_nearest"] = _time(
        lambda: sample_lookup(images, table, strategy="offset",
                              level_policy="nearest", rng=rng,
                              offset_per_level=True),
        repeats=repeats)

    # Isolate the two halves of the lookup cost so the conclusion is not a
    # single opaque number: index generation versus the gather itself.
    rows["index_generation_only_per_pixel"] = _time(
        lambda: (rng.integers(0, table.height, size=images.shape),
                 rng.integers(0, table.width, size=images.shape)),
        repeats=repeats)
    k_fixed = _level_index(images, table)
    u_fixed = rng.integers(0, table.height, size=images.shape)
    v_fixed = rng.integers(0, table.width, size=images.shape)
    rows["gather_only_per_pixel"] = _time(
        lambda: table.values[k_fixed, u_fixed, v_fixed].astype(np.float32),
        repeats=repeats)

    if include_jax:
        import jax
        import jax.random as jr
        from dluxshera.ml.noise import NoiseConfig, apply_pair_noise, apply_pair_noise_batch

        config = NoiseConfig(enabled=True, apply_to="observation",
                             noise_model="shera_observation", photon_noise=True,
                             bright_threshold=100.0, seed=12030,
                             negative_policy="clip")
        pairs = [(images[i], images[i]) for i in range(batch)]
        records = [(f"lookup-bench-{i}", 0) for i in range(batch)]
        rows["jax_reference_per_image"] = _time(
            lambda: [apply_pair_noise(a, b, config, pair_record_id=r,
                                      dynamic_seed_offset=o)
                     for (a, b), (r, o) in zip(pairs, records)],
            repeats=max(1, repeats // 2))
        rows["jax_batched"] = _time(
            lambda: apply_pair_noise_batch(pairs, config, records),
            repeats=max(1, repeats // 2))
        rows["jax_backend"] = {"backend": jax.default_backend(),
                               "jax_version": jax.__version__}

    baseline = rows["jax_batched"]["warm_seconds"] if include_jax else None
    for name, row in rows.items():
        if "warm_seconds" not in row:
            continue
        row["images_per_second"] = batch / row["warm_seconds"]
        row["seconds_per_32_image_batch"] = row["warm_seconds"] * (32.0 / batch)
        row["megapixels_per_second"] = pixels / row["warm_seconds"] / 1e6
        if baseline:
            row["speedup_vs_jax_batched"] = baseline / row["warm_seconds"]
    return {
        "batch_images": batch,
        "image_shape": [int(d) for d in images.shape[1:]],
        "repeats": int(repeats),
        "table": table.describe(),
        "level_coverage": level_coverage_report(images, table),
        "peak_rss_platform_units_self": resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss,
        "timings": rows,
        "note": (
            "Warm timings exclude table construction; build_seconds in "
            "table.describe() is the initialization cost. One noise view per "
            "image; E02 doubles the noise work."
        ),
    }


def memory_scaling_report(*, budget_mib: float) -> dict:
    """Analytical table-size scaling; allocates nothing."""
    rows = []
    for side in (64, 256, 1024):
        for levels in (256, 1024, 4096):
            for dtype in ("uint16", "float32"):
                nbytes = table_bytes(levels=levels, width=side, height=side, dtype=dtype)
                rows.append({
                    "spatial": [side, side],
                    "levels": levels,
                    "dtype": dtype,
                    "draws_per_level": side * side,
                    "mib": nbytes / 1024**2,
                    "gib": nbytes / 1024**3,
                    "within_default_budget": nbytes / 1024**2 <= budget_mib,
                    "four_worker_mib_if_copied": 4 * nbytes / 1024**2,
                })
    return {
        "default_budget_mib": float(budget_mib),
        "configurations": rows,
        "worker_sharing": (
            "DataLoader workers start via forkserver/spawn for S12, so a table "
            "built per worker is duplicated per worker. A read-only np.memmap on "
            "shared storage avoids that duplication, but then random per-pixel "
            "indexing becomes page-fault bound on a networked filesystem such as "
            "GPFS, which is the opposite of the intended optimization."
        ),
        "excluded": (
            "Figures cover table storage only: index arrays, gathered output, "
            "the clean image, and prefetched batches are additional."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_real_images(prepared_root: str | None, count: int) -> np.ndarray | None:
    if not prepared_root:
        return None
    from dluxshera.ml import load_sample_catalog
    catalog = load_sample_catalog(prepared_root)
    reader = catalog.image_reader(cache_size=4)
    total = int(catalog.sample_count)
    step = max(1, total // max(count, 1))
    return np.stack([reader.get(i * step % total) for i in range(count)])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--mode", default="all",
                        choices=("all", "performance", "fidelity", "correlation",
                                 "reuse", "memory", "exact"))
    parser.add_argument("--exact-delta", type=float, default=0.25,
                        help="Dyadic lattice resolution for the exact construction.")
    parser.add_argument("--table-side", type=int, default=256)
    parser.add_argument("--table-levels", type=int, default=1024,
                        help="Integer count levels 0..N-1 stored in the table.")
    parser.add_argument("--table-dtype", default="uint16",
                        choices=("uint16", "uint32", "float32"))
    parser.add_argument("--table-seed", type=int, default=20261031)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--draws", type=int, default=400_000)
    parser.add_argument("--images", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lambdas", type=float, nargs="+",
                        default=[0.1, 1.0, 5.0, 20.0, 60.0, 300.0])
    parser.add_argument("--prepared-root", default=None,
                        help="Optional prepared dataset for resident real images.")
    parser.add_argument("--no-jax", action="store_true")
    parser.add_argument("--memory-budget-mib", type=float,
                        default=DEFAULT_MEMORY_BUDGET_MIB,
                        help="Explicit opt-in required to allocate a larger table.")
    args = parser.parse_args(argv)

    requested = table_bytes(levels=args.table_levels, width=args.table_side,
                            height=args.table_side, dtype=args.table_dtype)
    if requested / 1024**2 > args.memory_budget_mib:
        parser.error(
            f"Requested table is {requested / 1024**2:.1f} MiB, above the "
            f"{args.memory_budget_mib:.1f} MiB budget. Raise "
            "--memory-budget-mib deliberately to allocate it."
        )

    out: dict = {"configuration": vars(args) | {"table_mib": requested / 1024**2}}
    if args.mode in ("all", "memory"):
        out["memory_scaling"] = memory_scaling_report(budget_mib=args.memory_budget_mib)

    table = PoissonTable(levels=integer_levels(args.table_levels - 1),
                         width=args.table_side, height=args.table_side,
                         dtype=args.table_dtype, seed=args.table_seed)
    out["table"] = table.describe()

    if args.mode in ("all", "performance"):
        real = _load_real_images(args.prepared_root, args.batch)
        synthetic = _synthetic_batch(args.batch, args.seed)
        out["performance_synthetic"] = performance_report(
            synthetic, table, repeats=args.repeats, seed=args.seed,
            include_jax=not args.no_jax)
        if real is not None:
            out["performance_real_images"] = performance_report(
                real, table, repeats=args.repeats, seed=args.seed,
                include_jax=not args.no_jax)
            out["performance_real_images"]["source"] = str(args.prepared_root)

    if args.mode in ("all", "fidelity"):
        out["fidelity"] = [
            distribution_report(lam, table, strategy=strategy,
                                level_policy=policy, draws=args.draws,
                                seed=args.seed)
            for lam in args.lambdas
            for strategy in INDEX_STRATEGIES
            for policy in LEVEL_POLICIES
        ]
        # Fractional means are where the two level policies diverge; measure
        # them explicitly rather than inferring from the integer grid.
        out["fidelity_fractional"] = [
            distribution_report(lam, table, strategy="per_pixel",
                                level_policy=policy, draws=args.draws,
                                seed=args.seed)
            for lam in (0.37, 1.5, 12.37)
            for policy in LEVEL_POLICIES
        ]

    if args.mode in ("all", "correlation"):
        out["correlation"] = [
            correlation_report(lam, table, strategy=strategy,
                               level_policy="nearest", images=args.images,
                               shape=IMAGE_SHAPE, seed=args.seed)
            for lam in (1.0, 20.0, 300.0)
            for strategy in INDEX_STRATEGIES
        ]

    if args.mode in ("all", "exact"):
        real = _load_real_images(args.prepared_root, args.batch)
        subject = real if real is not None else _synthetic_batch(args.batch, args.seed)
        out["exact_dyadic_construction"] = exact_construction_report(
            subject, delta=args.exact_delta, width=args.table_side,
            height=args.table_side, repeats=args.repeats, seed=args.seed)
        out["exact_dyadic_construction"]["images"] = (
            "resident real images" if real is not None else "synthetic PSF-like")

    if args.mode in ("all", "reuse"):
        out["reuse"] = [
            reuse_report(lam, table, strategy=strategy, images=args.images,
                         shape=IMAGE_SHAPE, seed=args.seed)
            for lam in (20.0,)
            for strategy in INDEX_STRATEGIES
        ]
        out["reproducibility"] = _reproducibility_report(table, args.seed)

    print(json.dumps(out, indent=2, sort_keys=True, default=float))
    return 0


def _synthetic_batch(batch: int, seed: int) -> np.ndarray:
    """Expected-count images shaped like a SHERA PSF: faint wings, bright core.

    A flat-lambda array would understate the level-index cost and overstate
    cache locality, so the synthetic batch spans the same several-decade count
    range the V3/V4 statistics show.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:IMAGE_SHAPE[0], 0:IMAGE_SHAPE[1]]
    out = np.empty((batch, *IMAGE_SHAPE), dtype=np.float32)
    for i in range(batch):
        cy, cx = rng.uniform(60, 100, size=2)
        r2 = (y - cy) ** 2 + (x - cx) ** 2
        out[i] = (2.0 + 900.0 * np.exp(-r2 / (2 * 9.0**2))
                  + 40.0 * np.exp(-r2 / (2 * 45.0**2)))
    return out


def _reproducibility_report(table: PoissonTable, seed: int) -> dict:
    """Confirm the prototype's own index schedule is seed-reproducible."""
    expected = np.full((4, 32, 32), 20.0)
    runs = [
        sample_lookup(expected, table, strategy=strategy, level_policy=policy,
                      rng=np.random.default_rng(seed))
        for strategy in INDEX_STRATEGIES
        for policy in LEVEL_POLICIES
        for _ in range(2)
    ]
    rebuilt = PoissonTable(levels=table.levels[:64], width=32, height=32,
                           dtype=table.dtype, seed=table.seed)
    again = PoissonTable(levels=table.levels[:64], width=32, height=32,
                         dtype=table.dtype, seed=table.seed)
    return {
        "same_seed_same_draws": all(
            np.array_equal(runs[i], runs[i + 1]) for i in range(0, len(runs), 2)),
        "same_seed_same_table": bool(np.array_equal(rebuilt.values, again.values)),
        "bitwise_identical_to_jax_prescription": False,
        "note": (
            "The prototype is internally reproducible, but a finite table "
            "cannot reproduce the prescribed JAX realization bitwise. Adopting "
            "it would require a new noise identity and a new frozen validation "
            "recipe, not a drop-in substitution."
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
