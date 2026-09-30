"""Optional Playwright check against independent pandas rolling statistics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[1]
with sync_playwright() as p:
 browser=p.chromium.launch()
 page=browser.new_page(viewport={'width':1440,'height':1100})
 errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto('http://127.0.0.1:8765/')
 page.select_option('#category','온도')
 page.fill('#featureSearch','Mold_Temperature_3')
 page.wait_for_function('document.querySelectorAll(".chart-card").length===1')
 def chart():
  page.locator('.time-plot').scroll_into_view_if_needed()
  page.wait_for_function('!!document.querySelector(".time-plot")?._fullLayout')
  return page.locator('.time-plot')
 for n in [5,15]:
  page.select_option('#temperatureWindow',str(n));c=chart()
  result=page.evaluate('''n=>{
   const d=JSON.parse(document.getElementById('embeddedData').textContent),i=d.features.findIndex(f=>f.name==='Mold_Temperature_3');
   const rows=d.records.filter(r=>r.product==='CN7'&&r.side==='LH');
   return {rows, i, actual:dashboardDiagnostics.rollingTemperature(rows,i,n),traces:document.querySelector('.time-plot').data.filter(t=>t.meta).map(t=>({kind:t.meta.kind,n:t.meta.windowSize}))};
  }''',n)
  rows=result['rows'];s=pd.Series([r['values'][result['i']] for r in rows]);times=pd.to_datetime([r['time'] for r in rows]);segments=pd.Series(times).diff().gt(pd.Timedelta(minutes=10)).cumsum()
  expected_m=s.groupby(segments).transform(lambda x:x.rolling(n).median())
  expected_s=s.groupby(segments).transform(lambda x:x.rolling(n).std(ddof=0))
  actual=result['actual'];keep=[i for i,t in enumerate(actual['x']) if t is not None]
  for name,expected in [('median',expected_m),('std',expected_s)]:
   values=[actual[name][i] for i in keep]
   np.testing.assert_allclose(np.array(values,dtype=float),expected.to_numpy(),atol=1e-9,equal_nan=True)
  assert len(result['traces'])==4 and all(t['n']==n for t in result['traces'])
 # Known synthetic data: no future influence, gap and missing-value reset.
 synthetic=[{'time':f'2020-01-01 00:0{i}:00','values':[v]} for i,v in enumerate([1,2,3,4,5])]
 synthetic += [{'time':'2020-01-01 01:00:00','values':[9]},{'time':'2020-01-01 01:01:00','values':[None]}]
 out=page.evaluate('r=>dashboardDiagnostics.rollingTemperature(r,0,5)',synthetic)
 assert out['median'][:5]==[None,None,None,None,3]
 assert abs(out['std'][4]-np.sqrt(2))<1e-12
 assert out['median'][5:]==[None,None,None]
 c.evaluate("e=>Plotly.relayout(e,{'xaxis.range':['2020-10-16 05:00:00','2020-10-16 06:00:00']})")
 axes=c.evaluate('e=>[e.layout.xaxis.range,e.layout.xaxis2.range,e.layout.xaxis3.range,e.layout.xaxis4.range]')
 assert all(x==axes[0] for x in axes)
 page.screenshot(path='/tmp/kamp-temperature-trends.png')
 page.select_option('#side','LH');c=chart();assert len(c.evaluate('e=>e.data.filter(t=>t.meta)'))==2
 page.select_option('#temperatureWindow','0');c=chart();assert not c.evaluate('e=>e.data.some(t=>t.meta)')
 assert not errors,errors
 report={'windows':[5,15],'pandas_comparison':'passed','gap_and_missing_reset':'passed','synchronized_four_time_axes':'passed','single_side_and_off_mode':'passed','page_errors':errors}
 (root/'artifacts/dashboard_v1/temperature_verification.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report));browser.close()
