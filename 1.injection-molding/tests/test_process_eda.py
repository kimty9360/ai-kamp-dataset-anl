"""Checks for temporal leakage and observation-gap semantics."""
import importlib.util
from pathlib import Path
import unittest
import pandas as pd

spec=importlib.util.spec_from_file_location('process_eda',Path(__file__).resolve().parents[1]/'scripts/process_eda.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class TemporalTests(unittest.TestCase):
    def dates(self):
        rows=[]
        for day in ['2020-01-01','2020-01-02','2020-01-04','2020-02-01','2020-02-03','2020-03-01']:
            for side in ['LH','RH']:
                rows.append({'product':'RG3','side':side,'date':day,'timestamp':pd.Timestamp(day),'defect':0})
        return pd.DataFrame(rows)
    def test_split_independent_of_labels_and_order(self):
        d=self.dates();a,r=m.assign_periods(d)
        d['defect']=1;d=d.sample(frac=1,random_state=42)
        b,s=m.assign_periods(d)
        self.assertEqual(r,s)
        self.assertEqual(a.set_index(['date','side']).period.to_dict(),b.set_index(['date','side']).period.to_dict())
    def test_paired_events_not_separated(self):
        d,_=m.assign_periods(self.dates())
        self.assertEqual(d.groupby('timestamp').period.nunique().max(),1)
        t=[d[d.period==p].timestamp for p in ['train','validation','test']]
        self.assertLess(t[0].max(),t[1].min());self.assertLess(t[1].max(),t[2].min())
    def test_three_dates_keep_nonempty_periods(self):
        d=self.dates();d=d[d.date.isin(sorted(d.date.unique())[:3])]
        d,_=m.assign_periods(d);self.assertEqual(set(d.period),{'train','validation','test'})
    def test_insufficient_dates_rejected(self):
        d=self.dates();d=d[d.date==d.date.min()]
        with self.assertRaises(ValueError):m.assign_periods(d)
    def test_gap_is_strictly_greater_than_threshold(self):
        d=pd.DataFrame({'product':['CN7']*4,'side':['RH']*4,'timestamp':pd.to_datetime(['2020-01-01 00:00','2020-01-01 00:10','2020-01-01 00:21','2020-01-01 00:22'])})
        g=m.segments(d)
        self.assertEqual(g.segment.tolist(),[1,1,2,2])
        self.assertTrue(pd.isna(g.gap_seconds.iloc[0]))
        self.assertEqual(g.minutes_since_observed_segment_start.tolist(),[0,10,0,1])
    def test_segments_do_not_cross_product_or_side(self):
        d=self.dates();g=m.segments(d)
        first=g.groupby(['product','side']).head(1)
        self.assertTrue(first.gap_seconds.isna().all())
        self.assertTrue(first.minutes_since_observed_segment_start.eq(0).all())

if __name__=='__main__':unittest.main()
