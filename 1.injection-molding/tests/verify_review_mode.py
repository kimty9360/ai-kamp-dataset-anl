"""Optional browser check for reversible record review filtering."""
import csv,io,json
from pathlib import Path
import pandas as pd
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[1]
with sync_playwright() as p:
 b=p.chromium.launch();page=b.new_page(accept_downloads=True);errors=[]
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto('http://127.0.0.1:8765/')
 assert page.locator('#reviewMode').input_value()=='include'
 assert page.locator('#recordCount').inner_text()=='3,974건'
 assert page.locator('#reviewTable tbody tr').count()==18
 page.select_option('#reviewMode','exclude')
 assert page.locator('#recordCount').inner_text()=='3,956건'
 assert page.locator('#defectCount').inner_text()=='28건'
 assert page.locator('#reviewTable tbody tr').count()==18
 page.locator('#reviewDetails summary').click();page.locator('#reviewTable tbody tr').first.click()
 assert '대상 (불량 재판정 아님)' in page.locator('#recordDetails').inner_text()
 page.click('#closeDialog')
 page.click('[data-tab="correlation"]')
 value=page.evaluate('dashboardDiagnostics.getCorrelation().matrix[0][1]')
 expected=pd.read_csv(root/'artifacts/cn7_oct29_30_audit/correlation_sensitivity.csv')
 expected=expected[(expected.cohort=='exclude_review_flag_only')&(expected.feature_a=='Injection_Time')].iloc[0].pearson
 assert abs(value-expected)<1e-10,(value,expected)
 with page.expect_download() as dl:page.click('#exportData')
 download=dl.value;rows=list(csv.DictReader(io.StringIO(Path(download.path()).read_text(encoding='utf-8-sig'))))
 assert len(rows)==3956 and all(r['Data_Review_Flag']=='0' for r in rows)
 assert download.suggested_filename.endswith('_exclude.csv')
 page.select_option('#product','RG3');assert page.locator('#recordCount').inner_text()=='1,256건'
 assert page.locator('#reviewDetails').is_hidden()
 page.select_option('#product','CN7');page.fill('#start','2020-10-29');page.dispatch_event('#start','change');page.fill('#end','2020-10-30');page.dispatch_event('#end','change')
 assert page.locator('#recordCount').inner_text()=='1,191건'
 page.select_option('#reviewMode','include');assert page.locator('#recordCount').inner_text()=='1,209건'
 assert not errors,errors
 result={'counts_and_labels':'passed','pearson_matches_audit':value,'csv_exclusion':'passed','excluded_record_details':'passed','date_and_product_filters':'passed','page_errors':errors}
 (root/'artifacts/dashboard_v1/review_mode_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result);b.close()
