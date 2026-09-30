import sys
import unittest
from pathlib import Path
import pandas as pd
from pandas.testing import assert_frame_equal
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from improve_v1 import transform, VARIANTS

class FeatureTests(unittest.TestCase):
    def test_duplicate_inputs_remain_identical_without_mutating_original(self):
        row={f'Barrel_Temperature_{i}':i/10 for i in range(1,7)}
        row.update(Mold_Temperature_3=1.,Mold_Temperature_4=-1.,Max_Back_Pressure=0.,Average_Back_Pressure=-2.)
        x=pd.DataFrame([row,row]); original=x.copy(deep=True)
        for variant in VARIANTS:
            out=transform(x,variant)
            assert_frame_equal(x,original)
            self.assertTrue(out.iloc[0].equals(out.iloc[1]))
            self.assertNotIn('PassOrFail',out)
        self.assertEqual(transform(x,'temperature_differences').shape[1],x.shape[1]+6)
        self.assertEqual(transform(x,'pressure_difference').shape[1],x.shape[1]+1)

    def test_unknown_variant_is_rejected(self):
        with self.assertRaises(ValueError): transform(pd.DataFrame({'x':[1]}),'typo')

if __name__=='__main__':unittest.main()
