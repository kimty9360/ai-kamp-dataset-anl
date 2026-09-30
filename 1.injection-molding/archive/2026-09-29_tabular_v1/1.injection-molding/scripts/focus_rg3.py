"""RG3 diagnosis, bounded raw-feature experiments, and inner Recall>=.95 policy."""
import argparse,copy,json,shutil,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score,precision_recall_curve,roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from sklearn.base import BaseEstimator,ClassifierMixin
from xgboost import XGBClassifier
from threadpoolctl import threadpool_limits
from train_baseline import PROJECT,sha,make_model,probability,select_threshold,metrics

VARIANTS=['reference','raw','raw_engineered','raw_engineered_tuned','raw_engineered_weight']

def recall_threshold(y,p,target=.95):
    y=np.asarray(y);p=np.asarray(p)
    if not 0<target<=1 or set(np.unique(y))!={0,1} or not np.isfinite(p).all():raise ValueError('Invalid target/data')
    prec,rec,thresholds=precision_recall_curve(y,p)
    eligible=np.flatnonzero(rec[:-1]>=target)
    # Max precision; ties prefer highest threshold (smaller workload at same precision).
    best=eligible[np.flatnonzero(prec[eligible]==prec[eligible].max())[-1]]
    return float(thresholds[best]),float(prec[best]),float(rec[best])

def engineering(x):
    x=x.copy()
    x['mold_temperature_difference']=x.Mold_Temperature_3-x.Mold_Temperature_4
    barrel=x[[f'Barrel_Temperature_{i}' for i in range(1,7)]]
    x['barrel_temperature_range']=barrel.max(axis=1)-barrel.min(axis=1)
    x['barrel_temperature_std']=barrel.std(axis=1,ddof=0)
    for i in range(1,6):x[f'barrel_difference_{i}_{i+1}']=x[f'Barrel_Temperature_{i}']-x[f'Barrel_Temperature_{i+1}']
    x['back_pressure_difference']=x.Max_Back_Pressure-x.Average_Back_Pressure
    x['screw_speed_difference']=x.Max_Screw_RPM-x.Average_Screw_RPM
    return x

class SVMScore(ClassifierMixin,BaseEstimator):
    def __init__(self,C=1.,class_weight=None):self.C=C;self.class_weight=class_weight
    def fit(self,x,y):
        self.model_=SVC(C=self.C,kernel='rbf',gamma='scale',class_weight=self.class_weight,probability=False,random_state=42).fit(x,y)
        self.classes_=self.model_.classes_;return self
    def predict_proba(self,x):
        p=expit(self.model_.decision_function(x));return np.column_stack([1-p,p])

def factory(name,variant,cfg,y):
    if name=='catboost_balanced':return make_model(name,cfg,y)
    if name=='svm_rbf':return Pipeline([('variance',VarianceThreshold()),('scale',StandardScaler()),('model',SVMScore(C=3. if variant=='raw_engineered_tuned' else 1.,class_weight='balanced' if variant=='raw_engineered_weight' else None))])
    cc=copy.deepcopy(cfg['xgboost'])
    if variant=='raw_engineered_tuned':cc.update(max_depth=2,min_child_weight=10,reg_lambda=10.)
    weight=float((y==0).sum()/(y==1).sum())
    if variant=='raw_engineered_weight':weight=np.sqrt(weight)
    return Pipeline([('variance',VarianceThreshold()),('model',XGBClassifier(**cc,random_state=cfg['model_seed'],scale_pos_weight=weight))])

def diagnose(source,x,old_predictions,out):
    rg=source.copy();rg['side']=np.where(x.part_is_rh.eq(1),'RH','LH');rg['date']=rg.TimeStamp.str[:10]
    rg.groupby(['date','side']).PassOrFail.agg(rows='size',defects='sum').reset_index().to_csv(out/'daily_counts.csv',index=False)
    rg.groupby(['side','PassOrFail','Reason'],dropna=False).size().rename('rows').reset_index().to_csv(out/'defect_reasons.csv',index=False)
    right=rg.side.eq('RH');features=[]
    for c in x:
        if c=='part_is_rh':continue
        auc=roc_auc_score(rg.loc[right,'PassOrFail'],x.loc[right,c])
        features.append(dict(feature=c,direction_free_auc=max(auc,1-auc),good_mean=x.loc[right&rg.PassOrFail.eq(0),c].mean(),defect_mean=x.loc[right&rg.PassOrFail.eq(1),c].mean()))
    pd.DataFrame(features).sort_values('direction_free_auc',ascending=False).to_csv(out/'rh_feature_summary.csv',index=False)
    cohorts=[]
    for name,op in old_predictions.items():
        o=op[(op.scenario=='rg3')&(op.seed==42)&(op.model==name)].sort_values('row_position')
        np.testing.assert_array_equal(o.row_position,np.arange(len(rg)))
        scorecol='score' if 'score' in o else 'p_class_1'
        joined=rg[['TimeStamp','Reason','PassOrFail','side']].copy();joined['score']=o[scorecol].to_numpy();joined['pred']=o.prediction_inner_f1.to_numpy()
        selected=np.zeros(len(rg),bool)
        for _,fold in o.groupby('fold'):
            order=np.lexsort((fold.tie_key.to_numpy(),-fold[scorecol].to_numpy()))
            selected[fold.row_position.to_numpy()[order[:int(np.ceil(len(fold)*.1))]]]=True
        joined['top10']=selected
        for reason,part in joined[joined.PassOrFail.eq(1)].groupby('Reason'):
            cohorts.append(dict(model=name,reason=reason,defects=len(part),found_inner_f1=int(part.pred.sum()),found_top10=int(part.top10.sum()),mean_score=part.score.mean()))
    pd.DataFrame(cohorts).to_csv(out/'reference_errors_by_reason_seed42.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    d=rg[right].groupby('date').PassOrFail.agg(['size','sum'])
    axes[0].bar(d.index,d['sum']);axes[0].set_title('RG3 RH | defect count by production date');axes[0].tick_params(axis='x',rotation=20)
    top=pd.DataFrame(features).sort_values('direction_free_auc',ascending=False).head(8)
    axes[1].barh(top.feature.iloc[::-1],top.direction_free_auc.iloc[::-1]);axes[1].set_xlim(.5,1);axes[1].set_title('RH-only descriptive single-feature AUC (not CV)')
    fig.tight_layout();fig.savefig(out/'diagnosis.png',dpi=150);plt.close(fig)

def main(out):
    if out.exists():raise FileExistsError(out)
    base=PROJECT/'artifacts/baseline_v2';archive=PROJECT/'04. Dataset_Molding.zip'
    cfg=json.loads((base/'config.json').read_text());splits=json.loads((base/'splits.json').read_text());features=json.loads((base/'preflight.json').read_text())['retained_features']
    protected={str(p):sha(p) for dirname in ['baseline_v2','side_feature_v1','extra_models_v1'] for p in (PROJECT/'artifacts'/dirname).rglob('*') if p.is_file()}
    protected.update({str(p):sha(p) for p in (PROJECT/'dataset').glob('moldset_*.csv')})
    source_protocol=json.loads((PROJECT/'artifacts/side_feature_v1/protocol.json').read_text())
    assert sha(archive)==source_protocol['source_archive_sha256']
    for path,h in source_protocol['raw_and_baseline_sha256'].items():assert sha(Path(path))==h
    prep={}
    with zipfile.ZipFile(archive) as z:
        r=pd.read_csv(z.open('dataset/moldset_labeled.csv'))
        for g in ['cn7','rg3']:
            df=pd.read_csv(PROJECT/'dataset'/f'moldset_labeled_{g}.csv');source=r[r.PART_NAME.str.startswith(g.upper())].iloc[:len(df)].reset_index(drop=True)
            cols=[c for c in df if c not in ['Unnamed: 0','PassOrFail']]
            np.testing.assert_allclose(StandardScaler().fit_transform(source[cols]),df[cols],rtol=0,atol=1e-10);np.testing.assert_array_equal(source.PassOrFail,df.PassOrFail)
            side=source.PART_NAME.str.extract('(LH|RH)$',expand=False);assert side.isin(['LH','RH']).all()
            x=df[features].assign(part_is_rh=side.eq('RH').astype(int));raw=source[features].assign(part_is_rh=side.eq('RH').astype(int))
            rows=pd.read_csv(base/f'{g}_rows.csv');np.testing.assert_array_equal(rows.row_position,np.arange(len(df)))
            np.testing.assert_array_equal(rows.PassOrFail,df.PassOrFail)
            prep[g]=(source,x,raw,rows,df.PassOrFail.to_numpy())
    side_oof=pd.read_csv(PROJECT/'artifacts/side_feature_v1/oof_predictions.csv');extra_oof=pd.read_csv(PROJECT/'artifacts/extra_models_v1/oof_predictions.csv')
    out.mkdir(parents=True);(out/'rows').mkdir()
    protocol={'target_recall':.95,'variants':VARIANTS,'rg3_models':['xgboost_balanced','svm_rbf'],'cn7_model':'catboost_balanced reference only','input_changes':'same original 23 process features + RH; raw restoration then 10 physical-difference features; no time/Reason input','xgb_tuned':{'max_depth':2,'min_child_weight':10,'reg_lambda':10},'svm_tuned':{'C':3},'weight_variant':'XGB sqrt(neg/pos); SVM balanced; train-fold-only weights','selection':'inner OOF precision at Recall>=.95; ties inner AP then variant order; external labels never used','threshold_policies':['fixed_0.5','inner_f1','inner_recall95'],'sampling':'none; weights only','groups':'unchanged baseline process groups plus same-equipment/time disjoint check','validation_limit':'existing repeated group CV, NOT independent future/time evaluation','code_sha256':sha(Path(__file__)),'baseline_helper_sha256':sha(PROJECT/'scripts/train_baseline.py'),'archive_sha256':sha(archive),'protected_files':protected}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));shutil.copyfile(__file__,out/'focus_rg3_snapshot.py')
    diagnose(prep['rg3'][0],prep['rg3'][2],{'xgboost_balanced':side_oof,'svm_rbf':extra_oof},out)
    scores=[];preds=[];selections=[];references=0
    for split in splits:
        g=split['scenario']
        if g not in prep:continue
        source,x,raw,rows,y=prep[g];tr,va=np.array(split['train_indices']),np.array(split['validation_indices'])
        times=source.EQUIP_CD.astype(str)+'|'+source.TimeStamp.astype(str)
        def check(a,b):
            assert not set(rows.feature_group.iloc[a])&set(rows.feature_group.iloc[b]);assert not set(times.iloc[a])&set(times.iloc[b])
        check(tr,va);inner=[]
        for ss in split['inner_splits_relative_to_train']:
            it,iv=np.array(ss['train']),np.array(ss['validation']);check(tr[it],tr[iv]);inner.append((it,iv))
        for name in (['catboost_balanced'] if g=='cn7' else ['xgboost_balanced','svm_rbf']):
            candidates=[]
            for variant in (['reference'] if g=='cn7' else VARIANTS):
                xx=x if variant=='reference' else raw if variant=='raw' else engineering(raw)
                ip=np.full(len(tr),np.nan)
                for it,iv in inner:
                    model=factory(name,variant,cfg,y[tr[it]]);model.fit(xx.iloc[tr[it]],y[tr[it]]);ip[iv]=probability(model,xx.iloc[tr[iv]])
                assert np.isfinite(ip).all()
                t95,prec95,rec95=recall_threshold(y[tr],ip);tf=select_threshold(y[tr],ip);iap=average_precision_score(y[tr],ip)
                model=factory(name,variant,cfg,y[tr]);model.fit(xx.iloc[tr],y[tr]);p=probability(model,xx.iloc[va])
                if variant=='reference':
                    old=extra_oof if name=='svm_rbf' else side_oof;old=old[(old.scenario==g)&(old.seed==split['seed'])&(old.fold==split['fold'])&(old.model==name)].sort_values('row_position')
                    np.testing.assert_array_equal(old.row_position,va)
                    np.testing.assert_allclose(p,old['score' if name=='svm_rbf' else 'p_class_1'],rtol=0,atol=1e-12)
                    np.testing.assert_allclose(tf,old.selected_threshold,rtol=0,atol=1e-12);references+=1
                candidates.append((prec95,iap,variant,p,tf,t95,rec95))
                common=dict(scenario=g,seed=split['seed'],fold=split['fold'],model=name,variant=variant,inner_precision95=prec95,inner_recall95=rec95,inner_ap=iap)
                for policy,t in [('fixed_0.5',.5),('inner_f1',tf),('inner_recall95',t95)]:
                    m=metrics(y[va],p,p>=t,rows.tie_key.to_numpy()[va],cfg['ranking_budgets'])
                    if name=='svm_rbf':m['brier']=np.nan;m['log_loss']=np.nan
                    scores.append({**common,'policy':policy,'threshold':t,**m,'inspection_fraction':float((p>=t).mean()),'recall95_met':m['recall']>=.95})
                preds.append(pd.DataFrame({**common,'row_position':va,'PassOrFail':y[va],'score':p,'threshold_f1':tf,'threshold95':t95,'tie_key':rows.tie_key.to_numpy()[va]}))
            if g=='rg3':
                best=max(candidates,key=lambda c:(c[0],c[1]));prec95,iap,v,p,tf,t95,rec95=best
                selections.append(dict(scenario=g,seed=split['seed'],fold=split['fold'],model=name,selected_variant=v,inner_precision95=prec95,inner_ap=iap))
                for policy,t in [('fixed_0.5',.5),('inner_f1',tf),('inner_recall95',t95)]:
                    m=metrics(y[va],p,p>=t,rows.tie_key.to_numpy()[va],cfg['ranking_budgets'])
                    if name=='svm_rbf':m['brier']=np.nan;m['log_loss']=np.nan
                    scores.append(dict(scenario=g,seed=split['seed'],fold=split['fold'],model=name,variant='inner_selected',inner_precision95=prec95,inner_recall95=rec95,inner_ap=iap,policy=policy,threshold=t,**m,inspection_fraction=float((p>=t).mean()),recall95_met=m['recall']>=.95))
            print('Completed',split['key'],name,flush=True)
    f=pd.DataFrame(scores);f.to_csv(out/'fold_metrics.csv',index=False)
    pd.concat(preds,ignore_index=True).to_csv(out/'oof_predictions.csv',index=False);pd.DataFrame(selections).to_csv(out/'selections.csv',index=False)
    keys=['ap','f1','precision','recall','inspection_fraction','recall_at_5pct','recall_at_10pct','recall_at_20pct','recall95_met']
    summary=f.groupby(['scenario','model','variant','policy'])[keys].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary=summary.reset_index();summary.to_csv(out/'summary.csv',index=False)
    # Per-repeat aggregate counts: repeated predictions are not additional independent products.
    rep=f.groupby(['scenario','model','variant','policy','seed'])[['tp','fp','fn','tn','rows','positives']].sum().reset_index()
    rep['precision']=rep.tp/(rep.tp+rep.fp);rep['recall']=rep.tp/(rep.tp+rep.fn);rep['inspection_fraction']=(rep.tp+rep.fp)/rep.rows
    rep.to_csv(out/'repeat_counts.csv',index=False)
    lines=['# RG3 집중 개선 및 Recall 95% 목표 평가','', 'CN7은 기존 좌우 CatBoost balanced를 유지하고 정책만 비교했다. RG3는 XGBoost balanced와 RBF SVM에서 원단위 입력, 물리 차이 변수, 작은 튜닝, 가중치 변경을 각각 고정 비교했다. 기존 외부/내부 그룹 분할을 유지했다. 입력은 원단위로 복구했으나 변수 수와 행은 그대로이며 새로운 결측/이상치 삭제는 하지 않았다.', '', 'RG3는 LH 591개 모두 양품, RH는 양품566·불량25개다. 불량 사유는 가스17·미성형8이다. 생산일은 10월21일 불량0, 22일15, 23일10이다. Reason과 시간은 진단용이며 모델 입력에서 제외했다. RH만의 단일 변수 분포/AUC도 전체 라벨 기반 탐색이므로 검증 성능이 아니다.', '', 'inner_recall95는 내부 OOF Recall>=.95를 만족하는 임계값 중 Precision 최대, 동률이면 높은 임계값을 고른다. inner_selected는 같은 내부 기준으로 후보를 선택한다. 외부 목표 달성은 보장되지 않는다. 외부 점수의 가장 좋은 후보를 사후 선택한 결과와 구분해야 한다.', '', '|데이터|모델|설정|정책|AP|Precision|Recall|F1|검사비율|Recall@10%|95%달성 폴드|','|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary[summary.policy.isin(['inner_f1','inner_recall95'])].itertuples():lines.append(f'|{r.scenario}|{r.model}|{r.variant}|{r.policy}|{r.ap_mean:.4f}|{r.precision_mean:.3f}|{r.recall_mean:.3f}|{r.f1_mean:.3f}|{r.inspection_fraction_mean:.3f}|{r.recall_at_10pct_mean:.3f}|{r.recall95_met_mean*9:.0f}/9|')
    lines+=['','summary.csvには固定0.5政策とRecall@5/10/20%も保存した。repeat_counts.csvは各seedで原本行を一回ずつ数えたTP/FP/FN/TN。9fold平均と全行集計は異なる。繰り返しで同じ不良を複数の独立不良と数えない。','','候補は事前に固定したが既存EDAを利用した開発比較であり独立テストではない。学習foldだけで前処理・重み・モデルをfitした。新たな生産データは追加していない。時間順検証や未来性能の保証はしていない。原単位差も実際のセンサ定義の限界がある。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    for path,h in protected.items():assert sha(Path(path))==h,path
    assert references==27
    (out/'verification.json').write_text(json.dumps({'status':'passed','reference_predictions_and_f1_thresholds_reproduced':references,'existing_files_unchanged':True,'old_groups_and_time_groups_disjoint':True,'new_outer_fits':99,'independent_test':False},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=4):main(a.output_dir)
