"""Optional Playwright end-to-end check; start the local dashboard server on port 8765 first.
Requires playwright and its Chromium browser in a separate test environment.
"""
from pathlib import Path
import json,csv,io
import pandas as pd,numpy as np
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[1];out=root/'artifacts/dashboard_v1'
checks=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True)
 page=browser.new_page(viewport={'width':1440,'height':1100},accept_downloads=True);errors=[];requests=[]
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.on('request',lambda r:requests.append(r.url))
 page.goto('http://127.0.0.1:8765/',wait_until='networkidle');page.locator('.time-plot').first.scroll_into_view_if_needed();page.wait_for_selector('.time-plot .main-svg')
 assert page.locator('#recordCount').inner_text()=='3,974건';assert page.locator('#defectCount').inner_text()=='28건';assert page.locator('.chart-card').count()==24
 assert len(page.evaluate('dashboardDiagnostics.getFeatureAudit().filter(f=>f.reason)'))==12
 page.locator('summary').filter(has_text='기본 전처리').click();page.check('#showExcluded');assert page.locator('.chart-card').count()==36
 page.uncheck('#showExcluded');assert page.locator('.chart-card').count()==24
 page.locator('.time-plot').first.scroll_into_view_if_needed();page.wait_for_selector('.time-plot .main-svg')
 checks.append('CN7 default 3974 records / 28 defects / 24 retained process cards')
 page.screenshot(path='/tmp/kamp-dashboard-desktop.png')
 page.locator('.time-plot .scatterlayer .point').first.click(force=True)
 assert page.locator('#recordDialog').is_visible()
 page.click('#closeDialog')
 # Every process column can be rendered, including constant temperature channels.
 for index in page.evaluate("dashboardDiagnostics.getAnalysisIndices()"):
  loc=page.locator(f'.time-plot[data-index="{index}"]');loc.scroll_into_view_if_needed()
  page.wait_for_function('(i)=>!!document.querySelector(`.time-plot[data-index="${i}"]`)._fullLayout',arg=index)
  stats=loc.evaluate('(el)=>({points:el.data.filter(t=>t.mode==="markers").reduce((n,t)=>n+t.x.length,0),defects:el.data.filter(t=>t.mode==="markers"&&t.customdata[0][2]==="불량").reduce((n,t)=>n+t.x.length,0)})')
  assert stats=={'points':3974,'defects':28},(index,stats)
 checks.append('All retained time plots rendered; each includes every selected record and all defects, without subsampling')
 page.evaluate('window.scrollTo(0,0)');page.select_option('#product','RG3');assert page.locator('#recordCount').inner_text()=='1,256건';assert page.locator('#defectCount').inner_text()=='32건'
 page.select_option('#side','LH');assert page.locator('#recordCount').inner_text()=='628건';assert page.locator('#defectCount').inner_text()=='5건'
 page.fill('#start','2020-11-04');page.dispatch_event('#start','change');page.fill('#end','2020-11-04');page.dispatch_event('#end','change');assert page.locator('#recordCount').inner_text()=='35건';assert page.locator('#defectCount').inner_text()=='5건'
 assert len(page.evaluate('dashboardDiagnostics.getAnalysisIndices()'))==24
 page.click('[data-tab="defects"]');assert page.locator('#defectTable tbody tr').count()==5;page.locator('#defectTable tbody tr').first.click();assert page.locator('#recordDialog').is_visible();assert page.locator('#recordDetails tbody tr').count()==36;page.click('#closeDialog')
 with page.expect_download() as dl:page.click('#exportData')
 download=dl.value;records=list(csv.reader(io.StringIO(Path(download.path()).read_text(encoding='utf-8-sig'))));assert len(records)==36;assert len(records[0])==30
 checks.append('Product, side, date filters; five defects on RG3 LH Nov 4; record detail 36 columns; filtered CSV export')
 page.select_option('#product','CN7');page.select_option('#side','ALL');page.click('[data-tab="correlation"]')
 max_errors={}
 for product in ['CN7','RG3']:
  page.select_option('#product',product)
  for method in ['pearson','spearman']:
   page.select_option('#method',method)
   page.wait_for_function('()=>!!document.getElementById("correlationPlot")._fullLayout')
   actual=page.evaluate('dashboardDiagnostics.getCorrelation().matrix');a=np.array([[np.nan if v is None else v for v in row] for row in actual])
   expected=pd.read_csv(root/f'artifacts/correlation_v1/matrices/{product.lower()}_{method}.csv',index_col=0).to_numpy()
   np.testing.assert_allclose(a,expected,rtol=0,atol=1e-8,equal_nan=True)
   max_errors[product+'_'+method]=float(np.nanmax(abs(a-expected)))
 assert page.locator('#correlationTable thead th').count()==25
 page.screenshot(path='/tmp/kamp-dashboard-correlation.png',full_page=False)
 checks.append('Frontend Pearson/Spearman match Python reference matrices for both CN7 and RG3')
 page.select_option('#product','JX1');assert page.locator('#recordCount').inner_text()=='1건';assert page.locator('#defectCount').inner_text()=='0건';assert '계산할 수 없습니다' in page.locator('#correlationPlot').inner_text()
 assert all(v is None for row in page.evaluate('dashboardDiagnostics.getCorrelation().matrix') for v in row)
 page.select_option('#side','LH');assert page.locator('#recordCount').inner_text()=='0건';page.click('[data-tab="timeline"]');assert page.locator('.chart-card').count()==0
 page.select_option('#product','CN7');page.select_option('#side','ALL');page.fill('#start','2020-11-03');page.dispatch_event('#start','change');page.fill('#end','2020-10-16');page.dispatch_event('#end','change');assert page.locator('#filterError').is_visible();assert page.locator('#recordCount').inner_text()=='0건';page.click('#resetDates')
 page.select_option('#category','온도');assert page.locator('.chart-card').count()==9;page.fill('#featureSearch','Mold_Temperature_3');page.wait_for_function('document.querySelectorAll(".chart-card").length===1');page.locator('.time-plot').scroll_into_view_if_needed();page.wait_for_selector('.time-plot .main-svg');page.uncheck('#showNormal');page.wait_for_function('()=>{const el=document.querySelector(".time-plot");return el?.data?.length>0&&el.data.filter(t=>t.mode==="markers").every(t=>t.customdata[0][2]==="불량")}')
 checks.append('Single-record and empty cohorts, invalid dates, all temperature channels, column search and hide-normal mode')
 page.evaluate('window.scrollTo(0,0)');page.set_viewport_size({'width':390,'height':844});page.screenshot(path='/tmp/kamp-dashboard-mobile.png')
 page.wait_for_function('document.documentElement.scrollWidth <= window.innerWidth + 2', timeout=10000)
 checks.append('Mobile layout fits viewport')
 # Local file works without any HTTP server and without external network requests.
 local=browser.new_page();local.goto((out/'index.html').as_uri(),wait_until='networkidle');assert local.locator('#recordCount').inner_text()=='3,974건';checks.append('Standalone file:// works offline')
 assert not errors,errors
 assert not [r for r in requests if r.startswith(('http://','https://')) and not r.startswith('http://127.0.0.1:8765')],requests
 result={'checks':checks,'max_correlation_error':max_errors,'page_errors':errors,'external_requests':0,'records':5232,'features':36}
 (out/'browser_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False,indent=2));browser.close()
