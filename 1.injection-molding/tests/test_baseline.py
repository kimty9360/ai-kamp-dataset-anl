import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('baseline',Path(__file__).resolve().parents[1]/'scripts/train_baseline.py')
b=importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class BaselineTests(unittest.TestCase):
    def test_conflicting_pairs_stay_together(self):
        groups=np.repeat(np.arange(30),2)
        y=np.tile([0,1],30)
        splits=b.group_splits(groups[:,None],y,groups,3,42)
        seen=[]
        for train,valid in splits:
            self.assertFalse(set(groups[train])&set(groups[valid]))
            seen.extend(valid)
        self.assertEqual(sorted(seen),list(range(60)))

    def test_threshold_matches_best_attainable_f1(self):
        y=np.array([0,1,0,1,0]); p=np.array([0.1,0.2,0.2,0.8,0.9])
        t=b.select_threshold(y,p)
        self.assertAlmostEqual(b.f1_score(y,p>=t),max(b.f1_score(y,p>=v) for v in np.unique(p)))

    def test_ranking_budget_rounding_and_confusion(self):
        y=np.array([1,0,0,1]); p=np.array([0.9,0.8,0.2,0.1]); tie=np.array([3,2,1,0])
        m=b.metrics(y,p,p>=0.5,tie,[0.1,0.5])
        self.assertEqual((m['tp'],m['fp'],m['fn'],m['tn']),(1,1,1,1))
        self.assertEqual(m['recall_at_10pct'],0.5)
        self.assertEqual(m['precision_at_10pct'],1.0)
        self.assertEqual(m['recall_at_50pct'],0.5)

    def test_tie_order_does_not_consult_labels(self):
        y=np.array([1,0,0,1]); p=np.full(4,0.1); tie=np.array([3,0,1,2])
        m=b.metrics(y,p,p>=0.5,tie,[0.5])
        self.assertEqual(m['recall_at_50pct'],0.0)


if __name__=='__main__': unittest.main()
