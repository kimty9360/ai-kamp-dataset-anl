import importlib.util
from pathlib import Path
import unittest
import pandas as pd

spec = importlib.util.spec_from_file_location('data_audit', Path(__file__).resolve().parents[1] / 'scripts/data_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def fixture(self):
        df = pd.DataFrame({f'x{i}': [float(g) for g in range(12) for _ in range(2)] for i in range(24)})
        df.insert(0, 'PassOrFail', [0, 1] * 12)
        df.insert(0, 'Unnamed: 0', range(24))
        return df

    def test_conflicts_ignore_id_and_label_and_keep_input(self):
        df = self.fixture()
        before = df.copy(deep=True)
        result, membership = audit.audit_frame(df, True)
        pd.testing.assert_frame_equal(df, before)
        self.assertEqual(result['duplicate_full_rows'], 0)
        self.assertEqual(result['unique_feature_groups'], 12)
        self.assertEqual(result['conflicting_groups'], 12)
        self.assertEqual(result['rows_in_conflicting_groups'], 24)
        self.assertEqual(membership.feature_group.iloc[0], membership.feature_group.iloc[1])
        self.assertTrue(all(f['overlapping_groups'] == 0 for f in result['diagnostic_3fold_seed42']))

    def test_consistent_duplicates_are_not_conflicts(self):
        df = self.fixture()
        df['PassOrFail'] = [g % 2 for g in range(12) for _ in range(2)]
        result, _ = audit.audit_frame(df, True)
        self.assertEqual(result['conflicting_groups'], 0)
        self.assertEqual(result['duplicate_feature_rows_beyond_first'], 12)

    def test_invalid_label_rejected(self):
        df = self.fixture()
        df.loc[0, 'PassOrFail'] = 2
        with self.assertRaises(ValueError):
            audit.audit_frame(df, True)


if __name__ == '__main__':
    unittest.main()
