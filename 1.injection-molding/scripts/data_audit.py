"""Read-only CSV audit. Group identifiers are local to each source file."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT = Path(__file__).resolve().parents[1]
FILES = [f'moldset_{kind}_{group}.csv' for kind in ('labeled', 'unlabeled') for group in ('cn7', 'rg3')]
ID, LABEL = 'Unnamed: 0', 'PassOrFail'


def audit_frame(df, labeled):
    if ID not in df or (LABEL in df) != labeled:
        raise ValueError('Unexpected ID/label schema')
    features = [c for c in df if c not in (ID, LABEL)]
    if len(features) != 24 or not all(pd.api.types.is_numeric_dtype(df[c]) for c in features):
        raise ValueError('Expected 24 numeric process features')
    if labeled and (df[LABEL].isna().any() or set(df[LABEL].unique()) != {0, 1}):
        raise ValueError('Expected nonmissing binary labels 0 and 1')
    x = df[features]
    # Exact equality on process features only: never include ID or label.
    groups = x.groupby(features, dropna=False, sort=True).ngroup()
    sizes = groups.value_counts()
    numeric = df.select_dtypes(include='number')
    result = {
        'rows': len(df), 'columns': list(df), 'features': features,
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': int(df.isna().sum().sum()),
        'infinite_values': int(np.isinf(numeric).sum().sum()),
        'duplicate_full_rows': int(df.duplicated().sum()),
        'duplicate_feature_rows_beyond_first': int(x.duplicated().sum()),
        'unique_feature_groups': int(len(sizes)),
        'group_size_histogram': {str(k): int(v) for k, v in sizes.value_counts().sort_index().items()},
        'constant_features': x.columns[x.nunique(dropna=False) <= 1].tolist(),
        'id_unique': bool(df[ID].is_unique),
        'id_monotonic_increasing': bool(df[ID].is_monotonic_increasing),
        'feature_statistics': {},
    }
    for col in features:
        s = x[col]
        result['feature_statistics'][col] = {
            'mean': float(s.mean()), 'std_ddof0': float(s.std(ddof=0)),
            'std_ddof1': float(s.std(ddof=1)), 'min': float(s.min()),
            'max': float(s.max()), 'unique_values': int(s.nunique()),
        }
    membership = pd.DataFrame({'row_position': np.arange(len(df)), 'record_id': df[ID], 'feature_group': groups})
    if labeled:
        membership[LABEL] = df[LABEL]
        grouped = membership.groupby('feature_group')[LABEL]
        conflicts = grouped.nunique() > 1
        membership['label_conflict'] = groups.isin(conflicts.index[conflicts])
        result.update(
            label_counts={str(k): int(v) for k, v in df[LABEL].value_counts().sort_index().items()},
            conflicting_groups=int(conflicts.sum()),
            rows_in_conflicting_groups=int(membership['label_conflict'].sum()),
            groups_containing_class_1=int((grouped.max() == 1).sum()),
        )
        # Feasibility diagnostic, not the final model-selection protocol.
        if result['missing_values'] == 0 and result['infinite_values'] == 0:
            folds = []
            cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
            for fold, (train, valid) in enumerate(cv.split(x, df[LABEL], groups)):
                overlap = set(groups.iloc[train]) & set(groups.iloc[valid])
                if overlap:
                    raise AssertionError('Feature groups cross split boundary')
                folds.append({'fold': fold, 'train_rows': len(train), 'valid_rows': len(valid),
                              'train_class_1': int(df[LABEL].iloc[train].sum()),
                              'valid_class_1': int(df[LABEL].iloc[valid].sum()), 'overlapping_groups': len(overlap)})
            result['diagnostic_3fold_seed42'] = folds
    return result, membership


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run(data_dir, output_dir):
    report = {'audit_version': '1.0', 'python': platform.python_version(),
              'numpy': np.__version__, 'pandas': pd.__version__,
              'group_definition': 'Exact equality of 24 process features within each file; ID and label excluded.',
              'limitations': ['Label semantics unconfirmed', 'No near-duplicate or cross-file matching',
                             'Diagnostic splits are not final validation splits', 'Not a complete Phase 1 distribution/EDA audit'],
              'files': {}}
    memberships = []
    expected_features = None
    for name in FILES:
        path = data_dir / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        stats, membership = audit_frame(pd.read_csv(path), '_labeled_' in name)
        if expected_features is None:
            expected_features = stats['features']
        if stats['features'] != expected_features:
            raise ValueError(f'Feature schema/order mismatch: {name}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Input changed during audit: {name}')
        stats.update(sha256=digest, bytes=path.stat().st_size)
        report['files'][name] = stats
        if '_labeled_' in name:
            membership.insert(0, 'source_file', name)
            memberships.append(membership)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'data_audit.json').write_text(json.dumps(clean_json(report), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    pd.concat(memberships, ignore_index=True).to_csv(output_dir / 'labeled_feature_groups.csv', index=False)
    lines = ['# 데이터 진단 결과', '', '라벨 의미는 미확인입니다. 그룹은 파일 내 24개 공정 특성의 정확한 일치로 정의합니다.', '',
             '| 파일 | 행 | 고유 특성 그룹 | 중복 초과행 | 라벨 충돌 그룹 |', '|---|---:|---:|---:|---:|']
    for name, stats in report['files'].items():
        lines.append(f"| {name} | {stats['rows']} | {stats['unique_feature_groups']} | {stats['duplicate_feature_rows_beyond_first']} | {stats.get('conflicting_groups', '해당 없음')} |")
    lines += ['', '원본은 수정하지 않습니다. SHA-256은 JSON에, 라벨 행별 그룹과 충돌 여부는 CSV에 저장했습니다.',
              '3-fold 분할은 그룹 겹침 여부를 확인하는 진단이며 최종 검증 분할이 아닙니다.',
              '근접중복·파일 간 중복·분포 비교·공정 해석은 후속 진단 대상입니다.']
    (output_dir / 'data_audit.md').write_text('\n'.join(lines) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=PROJECT / 'dataset')
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'artifacts' / 'audit')
    args = parser.parse_args()
    if args.output_dir.resolve().is_relative_to(args.data_dir.resolve()):
        parser.error('Audit output must be outside the raw dataset directory')
    report = run(args.data_dir, args.output_dir)
    for name, stats in report['files'].items():
        print(name, stats['rows'], 'rows;', stats['unique_feature_groups'], 'groups;', stats.get('conflicting_groups', 'N/A'), 'conflicts')
