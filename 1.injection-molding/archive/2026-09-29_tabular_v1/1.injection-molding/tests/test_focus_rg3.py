import sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from focus_rg3 import recall_threshold,engineering
class FocusTests(unittest.TestCase):
    def test_recall_constrained_threshold_matches_exhaustive_with_ties(self):
        y=np.array([1,0,1,0,1,0,0,1]);p=np.array([.9,.9,.6,.6,.2,.2,.1,.05])
        for target in [.5,.75,.95,1.]:
            t,precision,recall=recall_threshold(y,p,target)
            candidates=[]
            for cut in np.unique(p):
                pred=p>=cut;tp=y[pred].sum();r=tp/y.sum();pr=tp/pred.sum()
                if r>=target:candidates.append((pr,cut,r))
            expected=max(candidates,key=lambda c:(c[0],c[1]))
            self.assertEqual((precision,t,recall),expected)
    def test_invalid_target_or_single_class_rejected(self):
        with self.assertRaises(ValueError):recall_threshold([0,0],[.2,.3])
        with self.assertRaises(ValueError):recall_threshold([0,1],[.2,.3],1.1)
    def test_physical_features_preserve_original_and_duplicate_identity(self):
        row={f'Barrel_Temperature_{i}':260+i for i in range(1,7)}
        row.update(Mold_Temperature_3=25.,Mold_Temperature_4=27.,Max_Back_Pressure=35.,Average_Back_Pressure=50.,Max_Screw_RPM=30.,Average_Screw_RPM=290.,part_is_rh=1)
        x=pd.DataFrame([row,row]);saved=x.copy();z=engineering(x)
        assert_frame_equal(x,saved);self.assertTrue(z.iloc[0].equals(z.iloc[1]));self.assertEqual(z.shape[1],x.shape[1]+10)
        self.assertEqual(z.mold_temperature_difference.iloc[0],-2.)
        self.assertTrue(np.isfinite(z.to_numpy()).all())
if __name__=='__main__':unittest.main()
