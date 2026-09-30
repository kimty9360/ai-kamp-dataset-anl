"""Read-only, retrospective CN7 source-quality sensitivity audit; no correction/training."""
from pathlib import Path
import hashlib,json,zipfile
import pandas as pd
from review_sources import META
ROOT=Path(__file__).resolve().parents[1]
def main():
 source=ROOT/'sources/04. Dataset_Molding.zip';digest=hashlib.sha256(source.read_bytes()).hexdigest()
 with zipfile.ZipFile(source) as z:
  d=pd.read_csv(z.open('dataset/labeled_data.csv')).drop_duplicates('_id')
 c=d[d.PART_NAME.str.startswith('CN7')].sort_values(['TimeStamp','PART_NAME']).copy()
 features=[x for x in c if x not in META]
 c['date']=c.TimeStamp.str[:10];period=c.date.isin(['2020-10-29','2020-10-30'])
 # Dataset-specific review flag, NOT a validated process limit or defect label.
 flag=c.Clamp_Open_Position.lt(100)&c.Injection_Time.sub(c.Filling_Time).gt(6)
 assert flag.equals(c.Clamp_Open_Position.lt(100))
 assert flag.equals(c.Injection_Time.sub(c.Filling_Time).gt(6))
 out=ROOT/'artifacts/cn7_oct29_30_audit';out.mkdir(parents=True,exist_ok=True)
 c[flag].to_csv(out/'review_flagged_records.csv',index=False)
 c.groupby('date').agg(records=('_id','size'),defects=('PassOrFail',lambda x:x.eq('N').sum()),rpm_min=('Average_Screw_RPM','min'),rpm_median=('Average_Screw_RPM','median'),rpm_max=('Average_Screw_RPM','max'),max_rpm_median=('Max_Screw_RPM','median')).to_csv(out/'daily_summary.csv')
 stats=[]
 for name,x in [('other_dates',c[~period]),('oct29_30_unflagged',c[period&~flag]),('review_flagged',c[flag])]:
  for col in features:stats.append({'cohort':name,'feature':col,'records':len(x),'min':x[col].min(),'median':x[col].median(),'max':x[col].max()})
 pd.DataFrame(stats).to_csv(out/'feature_ranges.csv',index=False)
 pairs=[('Injection_Time','Filling_Time'),('Barrel_Temperature_1','Barrel_Temperature_2'),('Barrel_Temperature_1','Hopper_Temperature'),('Mold_Temperature_3','Mold_Temperature_4')]
 rows=[]
 for name,x in [('all_cn7',c),('exclude_review_flag_only',c[~flag]),('exclude_both_dates',c[~period])]:
  for a,b in pairs:rows.append({'cohort':name,'records':len(x),'defects':int(x.PassOrFail.eq('N').sum()),'feature_a':a,'feature_b':b,'pearson':x[a].corr(x[b]),'spearman':x[a].corr(x[b],method='spearman')})
 corr=pd.DataFrame(rows);corr.to_csv(out/'correlation_sensitivity.csv',index=False)
 summary={'cn7_records':len(c),'period_records':int(period.sum()),'flagged_records':int(flag.sum()),'flagged_timestamps':int(c[flag].TimeStamp.nunique()),'flagged_quality_labels':c[flag].PassOrFail.value_counts().to_dict(),'flagged_by_date':c[flag].date.value_counts().sort_index().to_dict(),'flag_outside_period':int((flag&~period).sum()),'source_sha256':digest,'source_unchanged':hashlib.sha256(source.read_bytes()).hexdigest()==digest,'rule':'CN7 Clamp_Open_Position < 100 AND Injection_Time - Filling_Time > 6; retrospective review flag only','limitations':['Not proof of corruption or defect','No raw upstream data for Oct 29/30: supplied unlabeled_data ends Oct 23','Sensitivity comparison uses entire labeled sample; not predictive validation','Average_Screw_RPM scale not corrected; requires channel definition and upstream logs']}
 (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 lines=['# CN7 10/29–10/30 데이터 품질 진단','',f"전체 CN7 {len(c)}건 중 해당 날짜 {period.sum()}건. 이 중 검토 대상 {flag.sum()}건/{c[flag].TimeStamp.nunique()}개 시각이며 전부 Y(양품) 기록이다. 원본·라벨·대시보드 계산은 변경하지 않았다.",'','## 동시 다변수 변화','', '검토 규칙은 `Clamp_Open_Position < 100` 및 `Injection_Time - Filling_Time > 6`이다. 이번 자료에서 분리된 값군을 표시하는 사후 탐색 규칙이지 공정 허용 범위나 불량 기준이 아니다. 두 조건은 같은 18건을 각각 식별한다.','', '|컬럼|해당 날짜 나머지 기록: 최소–최대|검토 18건: 최소–최대|','|---|---:|---:|']
 for col in ['Injection_Time','Filling_Time','Cycle_Time','Clamp_Open_Position','Plasticizing_Position','Barrel_Temperature_1','Hopper_Temperature']:
  a=c[period&~flag][col];b=c[flag][col];lines.append(f'|{col}|{a.min():.3f}–{a.max():.3f}|{b.min():.3f}–{b.max():.3f}|')
 lines+=['','예: 10/29 04:36:12에는 사출 11.48초·충전 3.36초·형개 위치 69.64·배럴 온도1 245.4가 기록되고, 8초 뒤 04:36:20에는 사출 9.55초·충전 4.43초·형개 위치 647.99·배럴 온도1 275.3이 같은 S14/CN7/LH·RH로 기록된다. 여러 값군이 짧은 간격으로 교차하는 현상은 조건 변화 외에 다른 금형/설비 기록 혼입·시각 또는 연결 오류도 점검해야 할 근거다. 실제 측정 시각/센서 위치가 미확인이라 물리적으로 불가능한 기록이라고 확정하지 않는다.','','## 상관 민감도','','|컬럼 쌍|전체 Pearson|18건 제외 Pearson|전체 Spearman|18건 제외 Spearman|','|---|---:|---:|---:|---:|']
 for a,b in pairs:
  x=corr[(corr.feature_a==a)&(corr.cohort=='all_cn7')].iloc[0];y=corr[(corr.feature_a==a)&(corr.cohort=='exclude_review_flag_only')].iloc[0]
  lines.append(f'|{a} / {b}|{x.pearson:.4f}|{y.pearson:.4f}|{x.spearman:.4f}|{y.spearman:.4f}|')
 lines+=['','## RPM 별도 점검','','Average_Screw_RPM은 10/27까지 중앙값 약 292.4–292.5, 10/29 이후(11/3 포함)는 약 29.2이다. Max_Screw_RPM은 약 30대다. 동일 단위/집계 구간의 최대·평균이라면 불일치하므로 단위/배율/컬럼 정의 확인이 필요하다. 단순 10배 보정은 실행하지 않는다.','','## 권장 처리','','- 날짜 전체 1,209건을 제거하지 않는다. 18건만 검토 플래그로 보존하며 포함/제외 결과를 병기한다. 이는 불량 재라벨링이 아니다.','- 시계열 시각화에서도 원본 표시와 검토 대상 제외 보기를 구분하고 제외 건수를 표시하는 방식이 적절하다. 현재 대시보드는 변경하지 않았다.','- 센서 국부 잡음으로 보고 보간/평활만 적용하지 않는다. 다른 조건의 값군이면 이동 통계에도 영향을 준다.','- 상위 수집 로그, 금형 ID/레시피, 샷 번호, 센서 주소/단위 변경 이력을 대조한다. 제공 unlabeled_data에는 해당 날짜가 없다.','- 검토 대상 제외/포함 및 Average_Screw_RPM 제외/원값 유지 실험을 분리한다. 모델 평가 전에 처리 규칙을 고정하고 과거/미래 기간을 구분한다.']
 (out/'report.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
