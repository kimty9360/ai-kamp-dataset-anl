"""Reproducible grouped baseline comparison; no final deployment model selection."""
import argparse
import hashlib
import json
import math
import platform
import shutil
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, log_loss, matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

PROJECT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_splits(x, y, groups, folds, seed):
    result = list(StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(x, y, groups))
    seen = np.zeros(len(y), dtype=int)
    for tr, va in result:
        if set(groups[tr]) & set(groups[va]):
            raise ValueError('Group leakage')
        if len(np.unique(y[tr])) != 2 or len(np.unique(y[va])) != 2:
            raise ValueError('A fold lacks a class; revise the protocol before fitting')
        seen[va] += 1
    if not np.all(seen == 1):
        raise ValueError('Invalid validation coverage')
    return result


def make_model(name, cfg, y):
    if name == 'dummy_prior':
        return DummyClassifier(strategy='prior')
    if name.startswith('logistic'):
        return Pipeline([('variance', VarianceThreshold()), ('scale', StandardScaler()),
                         ('model', LogisticRegression(**cfg['logistic'], random_state=cfg['model_seed'],
                          class_weight='balanced' if name.endswith('balanced') else None))])
    if name.startswith('xgboost'):
        weight = float((y == 0).sum() / (y == 1).sum()) if name.endswith('balanced') else 1.0
        return Pipeline([('variance', VarianceThreshold()),
                         ('model', XGBClassifier(**cfg['xgboost'], random_state=cfg['model_seed'], scale_pos_weight=weight))])
    raise ValueError(name)


def select_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    score = np.divide(2 * precision[:-1] * recall[:-1], precision[:-1] + recall[:-1],
                      out=np.zeros_like(precision[:-1]), where=(precision[:-1] + recall[:-1]) != 0)
    return float(thresholds[np.flatnonzero(score == score.max())[-1]])


def probability(model, x):
    p = model.predict_proba(x)[:, list(model.classes_).index(1)]
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid probability')
    return p


def metrics(y, p, prediction, tie, budgets):
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    result = dict(ap=float(average_precision_score(y, p)), roc_auc=float(roc_auc_score(y, p)),
                  brier=float(brier_score_loss(y, p)), log_loss=float(log_loss(y, p, labels=[0, 1])),
                  f1=float(f1_score(y, prediction, zero_division=0)), precision=float(precision_score(y, prediction, zero_division=0)),
                  recall=float(recall_score(y, prediction, zero_division=0)), mcc=float(matthews_corrcoef(y, prediction)),
                  tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), fnr=float(fn/(fn+tp)), fpr=float(fp/(fp+tn)),
                  rows=len(y), positives=int(y.sum()), unique_scores=int(len(np.unique(p))))
    order = np.lexsort((tie, -p))
    for budget in budgets:
        k = max(1, math.ceil(len(y) * budget))
        hits = int(y[order[:k]].sum())
        key = f'{int(budget*100)}pct'
        result[f'recall_at_{key}'] = hits / int(y.sum())
        result[f'precision_at_{key}'] = hits / k
        result[f'lift_at_{key}'] = (hits/k) / float(y.mean())
    return result


def main(cfg, out):
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite run: {out}')
    frames, hashes = [], {}
    process_features = None
    for group in ('cn7', 'rg3'):
        path = PROJECT / 'dataset' / f'moldset_labeled_{group}.csv'
        hashes[path.name] = sha(path)
        df = pd.read_csv(path)
        cols = [c for c in df if c not in ('Unnamed: 0', 'PassOrFail')]
        if process_features is None:
            process_features = cols
        if cols != process_features or len(cols) != 24:
            raise ValueError('Invalid feature schema/order')
        if set(df.PassOrFail.unique()) != {0, 1} or not np.isfinite(df[cols].to_numpy()).all():
            raise ValueError('Invalid labels or feature values')
        for col in cfg['drop_features']:
            if df[col].nunique() != 1:
                raise ValueError(f'Expected constant: {col}')
        df['product_group'] = group
        df['source_file'] = path.name
        df['row_position'] = np.arange(len(df))
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    features = [c for c in process_features if c not in cfg['drop_features']]
    # Conservative grouping across products, on all retained process inputs.
    combined['feature_group'] = combined.groupby(features, sort=True).ngroup()
    conflict = combined.groupby('feature_group').PassOrFail.nunique() > 1
    combined['label_conflict'] = combined.feature_group.isin(conflict.index[conflict])
    combined['tie_key'] = np.random.default_rng(cfg['tie_seed']).permutation(len(combined))
    unique_x = combined[features].drop_duplicates().to_numpy()
    distance = cKDTree(unique_x).query(unique_x, k=2)[0][:, 1]
    audit = {'cross_product_exact_groups': int((combined.groupby('feature_group').product_group.nunique()>1).sum()),
             'retained_features': features, 'unique_patterns': len(unique_x),
             'minimum_distance_between_unique_patterns': float(distance.min()),
             'patterns_with_neighbor_distance_below_1e_6': int((distance < 1e-6).sum()),
             'distance_space': 'Euclidean on provided standardized values; not physical units'}
    # Freeze all external and internal partitions before any training.
    prepared, splits, split_log = {}, {}, []
    for scenario in cfg['scenarios']:
        df = combined.copy() if scenario == 'pooled' else combined[combined.product_group == scenario].copy()
        df = df.reset_index(drop=True)
        x = df[features].copy()
        if scenario == 'pooled':
            x['product_is_rg3'] = (df.product_group == 'rg3').astype(int)
        y, groups = df.PassOrFail.to_numpy(), df.feature_group.to_numpy()
        prepared[scenario] = (df, x, y, groups)
        for seed in cfg['outer_seeds']:
            for fold, (tr, va) in enumerate(group_splits(x, y, groups, cfg['outer_folds'], seed)):
                inner = group_splits(x.iloc[tr], y[tr], groups[tr], cfg['inner_folds'], seed+1000+fold)
                key = f'{scenario}_{seed}_{fold}'
                splits[key] = (tr, va, inner)
                split_log.append({'key': key, 'scenario': scenario, 'seed': seed, 'fold': fold,
                    'train_indices': tr.tolist(), 'validation_indices': va.tolist(),
                    'train_class_1': int(y[tr].sum()), 'validation_class_1': int(y[va].sum()),
                    'inner_splits_relative_to_train': [{'train': a.tolist(), 'validation': b.tolist()} for a,b in inner]})
    out.mkdir(parents=True)
    (out/'models').mkdir()
    (out/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    (out/'splits.json').write_text(json.dumps(split_log, indent=2)+'\n')
    (out/'preflight.json').write_text(json.dumps(audit, indent=2)+'\n')
    manifest = {'input_sha256': hashes, 'code_sha256': sha(Path(__file__)), 'python': platform.python_version(),
                'packages': {name: version(name) for name in ('numpy','pandas','scipy','scikit-learn','xgboost','joblib')},
                'status': 'running', 'deployment_model': False}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copyfile(__file__, out/'train_baseline_snapshot.py')
    rows, predictions = [], []
    start = time.perf_counter()
    for scenario, (df, x, y, groups) in prepared.items():
        df[['source_file','row_position','Unnamed: 0','product_group','feature_group','PassOrFail','label_conflict','tie_key']].to_csv(out/f'{scenario}_rows.csv',index=False)
        for seed in cfg['outer_seeds']:
            for fold in range(cfg['outer_folds']):
                key = f'{scenario}_{seed}_{fold}'
                tr, va, inner = splits[key]
                for name in cfg['models']:
                    t0 = time.perf_counter()
                    threshold = 0.5
                    if name != 'dummy_prior':
                        inner_p = np.full(len(tr), np.nan)
                        for it, iv in inner:
                            model = make_model(name, cfg, y[tr[it]])
                            model.fit(x.iloc[tr[it]], y[tr[it]])
                            inner_p[iv] = probability(model, x.iloc[tr[iv]])
                        if not np.isfinite(inner_p).all():
                            raise AssertionError('Incomplete inner OOF')
                        threshold = select_threshold(y[tr], inner_p)
                    model = make_model(name, cfg, y[tr])
                    model.fit(x.iloc[tr], y[tr])
                    train_seconds = time.perf_counter()-t0
                    t1 = time.perf_counter()
                    p = probability(model, x.iloc[va])
                    predict_seconds = time.perf_counter()-t1
                    model_path = out/'models'/f'{key}_{name}.joblib'
                    joblib.dump({'model': model, 'threshold': threshold, 'features': list(x.columns), 'positive_label': 1}, model_path)
                    restored = joblib.load(model_path)
                    np.testing.assert_allclose(probability(restored['model'],x.iloc[va]),p,rtol=0,atol=1e-12)
                    common = {'scenario':scenario, 'seed':seed,'fold':fold,'model':name,
                              'selected_threshold':threshold,'train_seconds_including_inner_cv':train_seconds,'predict_seconds':predict_seconds}
                    for policy, cutoff in [('fixed_0.5',0.5),('inner_f1',threshold)]:
                        rows.append({**common,'policy':policy,**metrics(y[va],p,p>=cutoff,df.tie_key.to_numpy()[va],cfg['ranking_budgets'])})
                    pred = df.iloc[va][['source_file','row_position','Unnamed: 0','product_group','feature_group','PassOrFail','label_conflict','tie_key']].copy()
                    pred = pred.assign(scenario=scenario,seed=seed,fold=fold,model=name,p_class_1=p,
                                       threshold=threshold,prediction_fixed=(p>=0.5).astype(int),prediction_inner_f1=(p>=threshold).astype(int))
                    predictions.append(pred)
                print(f'Completed {key}',flush=True)
        pd.DataFrame(rows).to_csv(out/'fold_metrics.csv',index=False)
        pd.concat(predictions,ignore_index=True).to_csv(out/'oof_predictions.csv',index=False)
    fm = pd.DataFrame(rows)
    metric_names = ['ap','roc_auc','f1','precision','recall','mcc','brier','recall_at_5pct','recall_at_10pct','lift_at_10pct']
    summary = fm.groupby(['scenario','model','policy'])[metric_names].agg(['mean','std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(out/'summary.csv',index=False)
    oof = pd.concat(predictions,ignore_index=True)
    repeat_rows=[]
    for (scenario,seed,name), p in oof.groupby(['scenario','seed','model']):
        if p.duplicated(['source_file','row_position']).any() or len(p)!=len(prepared[scenario][0]):
            raise AssertionError('Invalid OOF coverage')
        for subgroup in ['all'] + (['cn7','rg3'] if scenario=='pooled' else []):
            part = p if subgroup=='all' else p[p.product_group==subgroup]
            for policy, pred_col in [('fixed_0.5','prediction_fixed'),('inner_f1','prediction_inner_f1')]:
                repeat_rows.append(dict(scenario=scenario,seed=seed,model=name,product_group=subgroup,policy=policy,
                    **metrics(part.PassOrFail.to_numpy(),part.p_class_1.to_numpy(),part[pred_col].to_numpy(),part.tie_key.to_numpy(),cfg['ranking_budgets'])))
    pd.DataFrame(repeat_rows).to_csv(out/'repeat_oof_metrics.csv',index=False)
    for name, digest in hashes.items():
        if sha(PROJECT/'dataset'/name)!=digest:
            raise AssertionError('Input changed')
    manifest.update(status='complete',elapsed_seconds=time.perf_counter()-start,
                    outer_models=len(predictions),oof_rows=len(oof),reload_check='all passed',raw_hash_check='passed')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(summary.to_string(),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=PROJECT/'configs/baseline_v1.json')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    if cfg['positive_label']!=1:
        raise ValueError('This exploratory implementation evaluates class_1 only')
    out=args.output_dir or PROJECT/'artifacts'/cfg['run_name']
    with threadpool_limits(limits=4):
        main(cfg,out)
