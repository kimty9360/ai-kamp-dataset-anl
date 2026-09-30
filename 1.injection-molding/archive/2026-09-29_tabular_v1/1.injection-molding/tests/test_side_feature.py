import sys,unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_side_feature import join_metadata
class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.rows=pd.DataFrame({'source_file':['cn7','rg3'],'row_position':[0,0],'PassOrFail':[0,1]})
        self.meta=self.rows.assign(side=['LH','RH'],TimeStamp=['a','b'],EQUIP_CD=['S14','S14']).iloc[::-1]
    def test_product_key_prevents_cross_product_row_collision(self):
        result=join_metadata(self.rows,self.meta)
        self.assertEqual(result.side.tolist(),['LH','RH'])
    def test_missing_duplicate_and_conflicting_metadata_rejected(self):
        bad=self.meta.copy();bad['PassOrFail']=0
        for m in [self.meta.iloc[:1],pd.concat([self.meta,self.meta]),bad]:
            with self.assertRaises(ValueError):join_metadata(self.rows,m)
if __name__=='__main__':unittest.main()
