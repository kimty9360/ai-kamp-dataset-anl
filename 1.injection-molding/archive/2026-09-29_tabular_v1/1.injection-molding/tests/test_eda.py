import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('eda',Path(__file__).resolve().parents[1]/'scripts/eda_analysis.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)


class EDATests(unittest.TestCase):
    def test_inspection_budget_is_per_fold_and_ties_ignore_labels(self):
        df=pd.DataFrame({'scenario':['cn7']*6,'seed':[42]*6,'fold':[0,0,0,1,1,1],
                         'model':['m']*6,'p_class_1':[.5,.5,.1,.2,.1,.01],
                         'tie_key':[2,1,0,0,1,2],'PassOrFail':[1,0,0,0,1,0]})
        result=e.add_inspection_selection(df,.1)
        self.assertEqual(result.index[result.selected_top10].tolist(),[1,3])
        self.assertNotIn('selected_top10',df)

    def test_empty_class_recall_is_undefined_not_zero(self):
        result=e.error_counts([0,0],[0,1])
        self.assertTrue(np.isnan(result['recall']))
        self.assertEqual(result['fp'],1)
        self.assertEqual(result['fpr'],.5)

    def test_oof_join_rejects_label_mismatch_and_duplicates(self):
        raw=pd.DataFrame({'source_file':['a','a'],'row_position':[0,1],'PassOrFail':[0,1],
            'Unnamed: 0':[0,1],'product_group':['cn7','cn7'],'feature_group':[0,0],'label_conflict':[True,True]})
        oof=raw.copy().assign(scenario='cn7',seed=42,model='m',p_class_1=[.1,.2])
        cfg={'scenarios':['cn7'],'outer_seeds':[42],'models':['m']}
        self.assertEqual(len(e.validate_oof(oof,raw,cfg)),2)
        altered=oof.copy();altered.loc[0,'PassOrFail']=1
        with self.assertRaisesRegex(ValueError,'mismatch'):e.validate_oof(altered,raw,cfg)
        with self.assertRaisesRegex(ValueError,'Duplicate OOF'):e.validate_oof(pd.concat([oof,oof]),raw,cfg)

    def test_constant_feature_has_no_rank_separation(self):
        self.assertEqual(e.descriptive_auc([0,1,0,1],[0,0,0,0]),.5)
        self.assertTrue(np.isnan(e.descriptive_auc([0,0],[1,2])))


if __name__=='__main__':unittest.main()
