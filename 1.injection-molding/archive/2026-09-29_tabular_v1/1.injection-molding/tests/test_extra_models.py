import sys,unittest,pickle,warnings
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_extra_models import MarginSVC,extra_model
class ExtraModelsTests(unittest.TestCase):
    def test_margin_score_matches_svc_decision_and_roundtrip(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(60,3));y=(x[:,0]>0.7).astype(int)
        m=MarginSVC('balanced').fit(x,y);p=m.predict_proba(x)
        self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(p.sum(axis=1),1)
        np.testing.assert_array_equal((p[:,1]>.5).astype(int),m.predict(x))
        np.testing.assert_allclose(p,pickle.loads(pickle.dumps(m)).predict_proba(x))
    def test_all_new_families_fit_with_constant_column_and_no_random_validation(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(40,4));x[:,3]=0;y=np.array([0]*30+[1]*10)
        cfg={'random_forest':{'n_estimators':5,'max_depth':8,'min_samples_leaf':3,'max_features':'sqrt','n_jobs':1}}
        for name in ['svm_rbf','svm_rbf_balanced','random_forest','random_forest_balanced','gaussian_nb','dnn_mlp']:
            m=extra_model(name,cfg)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');m.fit(x,y)
            p=m.predict_proba(x)
            self.assertEqual(p.shape,(40,2));self.assertTrue(np.isfinite(p).all())
            np.testing.assert_allclose(p.sum(axis=1),1)
            self.assertEqual(m.named_steps['variance'].get_support().tolist(),[True,True,True,False])
            if name=='dnn_mlp':self.assertFalse(m.named_steps['model'].early_stopping)
if __name__=='__main__':unittest.main()
