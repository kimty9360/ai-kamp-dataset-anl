import importlib.util
from pathlib import Path
import unittest
import json
import tempfile
import joblib
from unittest.mock import patch
import numpy as np
import pandas as pd

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

    def test_new_models_fit_predict_and_reload(self):
        cfg=json.loads((b.PROJECT/'configs/baseline_v2.json').read_text())
        cfg['extratrees']['n_estimators']=10
        cfg['catboost']['iterations']=10
        cfg['lightgbm']['n_estimators']=10
        rng=np.random.default_rng(42)
        x=pd.DataFrame(rng.normal(size=(120,4)),columns=list('abcd'))
        x['constant']=1.0
        y=(x.a>0.9).astype(int).to_numpy()
        for name in cfg['models'][5:]:
            with self.subTest(model=name), tempfile.TemporaryDirectory() as tmp:
                model=b.make_model(name,cfg,y)
                model.fit(x,y)
                p=b.probability(model,x)
                self.assertEqual(p.shape,(120,))
                self.assertGreater(np.ptp(p),0)
                path=Path(tmp)/'model.joblib'
                joblib.dump(model,path)
                np.testing.assert_allclose(p,b.probability(joblib.load(path),x),rtol=0,atol=1e-12)

    def test_reference_split_and_config_drift_rejected(self):
        cfg=json.loads((b.PROJECT/'configs/baseline_v2.json').read_text())
        base={k:v for k,v in cfg.items() if k!='reference_run'}
        with tempfile.TemporaryDirectory() as tmp:
            ref=Path(tmp)/'artifacts/baseline_v1'
            ref.mkdir(parents=True)
            for name,content in [('config.json',base),('manifest.json',{'input_sha256':{'a':'hash'}}),
                                 ('preflight.json',{'features':['a']}),('splits.json',[{'fold':0}])]:
                (ref/name).write_text(json.dumps(content))
            with patch.object(b,'PROJECT',Path(tmp)):
                b.verify_reference(cfg,{'a':'hash'},{'features':['a']},[{'fold':0}])
                with self.assertRaisesRegex(ValueError,'splits'):
                    b.verify_reference(cfg,{'a':'hash'},{'features':['a']},[{'fold':1}])
                changed={**cfg,'tie_seed':99}
                with self.assertRaisesRegex(ValueError,'tie_seed'):
                    b.verify_reference(changed,{'a':'hash'},{'features':['a']},[{'fold':0}])


if __name__=='__main__': unittest.main()
