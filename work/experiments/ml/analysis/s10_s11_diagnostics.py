#!/usr/bin/env python3
"""Read-only validation-only follow-up diagnostics for completed S10 / S11.

Run from the dluxshera-sandbox repository root with numpy installed:
  python work/experiments/ml/analysis/s10_s11_diagnostics.py

Uses saved prediction arrays and metrics only. Does not evaluate models or test data.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path('Results/hpc_imports/ml/gattaca2')
BASIS = ROOT / 'S10/s10_science_eigenbasis.json'


def load_run(study: str, exp: str, seed_index: int):
    ordinal = (11, 23, 47).index(seed_index) + 1
    run_id = f'{study}-{exp}-R{ordinal:03d}'
    matches = sorted((ROOT / study).rglob(f'{run_id}/run_manifest.json'))
    if len(matches) != 1:
        raise RuntimeError(f'{run_id}: expected one run manifest; found {len(matches)}')
    run_dir = matches[0].parent
    with np.load(run_dir / 'evaluation_predictions.npz', allow_pickle=False) as contents:
        arrays = {key: contents[key] for key in contents.files}
    metrics = json.loads((run_dir / 'metrics.json').read_text())
    manifest = json.loads((run_dir / 'run_manifest.json').read_text())
    if manifest['run_id'] != run_id or manifest.get('test_evaluated') is not False:
        raise RuntimeError(f'{run_id}: unexpected run/test status')
    return run_id, arrays, metrics, manifest


def mse_error(arrays):
    return (np.asarray(arrays['y_pred_z'], dtype=np.float64) -
            np.asarray(arrays['y_true_z'], dtype=np.float64))


def rmse(x):
    return float(np.sqrt(np.mean(np.square(x))))


def same_pairs(ref, alt):
    for field in ('pair_record_id', 'eval_slice', 'y_true_z'):
        if field not in ref or field not in alt:
            return False, f'{field} absent'
        if not np.array_equal(ref[field], alt[field]):
            return False, f'{field} differs'
    return True, ''


def main():
    if not BASIS.exists():
        raise FileNotFoundError(f'Missing frozen eigenbasis: {BASIS}; run at repo root')
    basis = json.loads(BASIS.read_text())
    labels = basis['parameter_labels']
    Q = np.asarray(basis['eigenvectors'], dtype=np.float64)
    sigma = np.asarray(basis['fisher_scales'], dtype=np.float64)
    eig = np.asarray(basis['eigenvalues'], dtype=np.float64)
    assert len(labels) == 20 and Q.shape == (20,20) and np.allclose(Q.T @ Q, np.eye(20), atol=1e-6)
    assert np.all(eig[:-1] >= eig[1:])
    sep_idx = labels.index('source.separation_as')
    sep_mas_per_z = sigma[sep_idx] * 1000.0
    Q_hi = Q[:, :14]
    Q_weak = Q[:, 14:]

    print('Separation squared overlap:', Q[sep_idx]**2)
    print('=== S10: per-seed eigenmode groups and hybrid (validation only) ===')
    print('Group RMSE: Fisher-scaled z; sep RMSE: mas. Lower is better.')
    print('E01=ordinary; E02=strong-weighted; E03=weak-weighted.')
    print('seed  exp      strong  middle    weak    global  separation')
    hybrid_rows = []
    weak_rows = []
    for seed in (11,23,47):
        loaded = {exp: load_run('S10', exp, seed) for exp in ('E01','E02','E03')}
        errors = {}
        for exp in ('E01','E02','E03'):
            rid, ar, metrics, manifest = loaded[exp]
            error = mse_error(ar)
            if error.shape[1] != len(labels):
                raise RuntimeError(f'{rid}: prediction dimensions disagree with basis')
            projected = error @ Q
            assert np.allclose(np.sum(error**2), np.sum(projected**2), rtol=1e-12)
            assert np.isclose(rmse(error), metrics['validation']['fisher_overall_rmse'], rtol=1e-12)
            strong, middle, weak = (rmse(projected[:, s]) for s in (slice(0,6),slice(6,14),slice(14,20)))
            print('Mode RMSE', seed, exp, np.sqrt(np.mean(projected**2, axis=0)))
            print(f'  {seed:1d}   {exp:<3} {strong:>9.2f} {middle:>7.2f} {weak:>7.2f} {rmse(error):>9.2f} {rmse(error[:,sep_idx])*sep_mas_per_z:>9.3f}')
            errors[exp] = (error, projected)
        weak_rows.append((seed,rmse(errors['E01'][1][:,14:]),rmse(errors['E02'][1][:,14:]),rmse(errors['E03'][1][:,14:])))
        a = loaded['E01'][1]
        b = loaded['E02'][1]
        aligned, reason = same_pairs(a,b)
        if not aligned:
            print(f'  Seed {seed} hybrid SKIPPED: E01/E02 frozen pair alignment not demonstrated ({reason})')
            continue
        p1 = np.asarray(a['y_pred_z'],dtype=np.float64)
        p2 = np.asarray(b['y_pred_z'],dtype=np.float64)
        truth = np.asarray(a['y_true_z'],dtype=np.float64)
        hybrid = (p2 @ Q_hi) @ Q_hi.T + (p1 @ Q_weak) @ Q_weak.T
        e_hybrid = hybrid - truth
        assert np.allclose(hybrid @ Q_hi,p2 @ Q_hi,rtol=1e-9,atol=1e-8)
        assert np.allclose(hybrid @ Q_weak,p1 @ Q_weak,rtol=1e-9,atol=1e-8)
        vals = (rmse(errors['E01'][0]),rmse(errors['E02'][0]),rmse(e_hybrid),
                rmse(errors['E01'][0][:,sep_idx])*sep_mas_per_z,
                rmse(errors['E02'][0][:,sep_idx])*sep_mas_per_z,
                rmse(e_hybrid[:,sep_idx])*sep_mas_per_z)
        hybrid_rows.append((seed, *vals))

        # Partition the separation-coordinate residual into informative / weak contributions.
        sep_hi = ((errors['E02'][0] @ Q_hi) @ Q_hi.T)[:, sep_idx] * sep_mas_per_z
        sep_weak = ((errors['E02'][0] @ Q_weak) @ Q_weak.T)[:, sep_idx] * sep_mas_per_z
        assert np.allclose(sep_hi + sep_weak, errors['E02'][0][:,sep_idx]*sep_mas_per_z, atol=1e-7)
        print(f'     E02 separation residual: modes 1-14 RMSE={rmse(sep_hi):.4f} mas; modes 15-20 RMSE={rmse(sep_weak):.6f} mas (covariance matters)')

    print('\nS10 weak-group RMSE by seed (ordinary / strong / weak weighted):')
    for row in weak_rows:
        print(f'  seed {row[0]}: {row[1]:.3f} / {row[2]:.3f} / {row[3]:.3f}; E03/E01={row[3]/row[1]:.4f}')

    print('\nS10 exploratory hybrid = modes 1-14 from E02 + modes 15-20 from E01:')
    if not hybrid_rows:
        print('  No matching validation records; DO NOT combine predictions without verified alignment.')
    else:
        print('seed  E01_global E02_global Hybrid_global  E01_sep_mas E02_sep_mas Hybrid_sep_mas')
        for seed,*vals in hybrid_rows:
            print(f'{seed:4d} {vals[0]:>10.3f} {vals[1]:>10.3f} {vals[2]:>13.3f} {vals[3]:>11.4f} {vals[4]:>11.4f} {vals[5]:>14.4f}')
        print('Arithmetic means across seeds (not pooled RMSE):')
        avg=np.mean(np.asarray([row[1:] for row in hybrid_rows],dtype=float),axis=0)
        print('  ', ' '.join(f'{v:.4f}' for v in avg))
        print('NOTE: post-hoc hybrid using frozen validation predictions; exploratory, not a deployable CNN or independent validation.')

    print('\n=== S11: stored consistency diagnostics per seed ===')
    print('Antisymmetry MSE evaluated in both groups; E01 identity is NOT evaluated and its 0.0 is a disabled-term sentinel.')
    print('seed exp  antisymmetry_mse  identity_mse (where evaluated)  val_fisher_rmse')
    values = {exp:[] for exp in ('E01','E02')}
    for seed in (11,23,47):
        for exp in ('E01','E02'):
            rid, arrays, metrics, _ = load_run('S11',exp,seed)
            c = metrics.get('validation',{}).get('consistency',{})
            anti=c.get('antisymmetry_mse')
            identity = c.get('identity_mse') if exp == 'E02' else None
            if anti is None or (exp == 'E02' and identity is None):
                raise RuntimeError(f'{rid}: missing expected consistency metric')
            value=rmse(mse_error(arrays))
            print(f'  {seed:1d}  {exp:<3} {float(anti):>17.4f}  {float(identity):>15.4f}' if identity is not None else f'  {seed:1d}  {exp:<3} {float(anti):>17.4f}  {"N/A":>15}',end='')
            print(f'  {value:>17.3f}')
            values[exp].append(float(anti))
    means={exp:float(np.mean(v)) for exp,v in values.items()}
    print(f'S11 antisymmetry MSE means: E01 {means["E01"]:.4f}, E02 {means["E02"]:.4f}; E02/E01 {means["E02"]/means["E01"]:.3f}')
    print('This does not establish identity improvement relative to E01 or antisymmetry vs the unregularized S10 reference.')
    print('\nDone. No files modified, no test evaluation, no checkpoints or network inference.')

if __name__=='__main__':
    try:
        main()
    except (Exception, AssertionError) as exc:
        print(f'DIAGNOSTIC STOP: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise
