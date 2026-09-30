"""Additional supervised model families on frozen side-feature inputs and splits."""
import argparse
import json
import shutil
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from train_baseline import PROJECT, sha, make_model, probability, select_threshold, metrics


def join_metadata(rows, metadata):
    keys=['source_file','row_position']
    if metadata.duplicated(keys).any(): raise ValueError('Duplicate metadata key')
    result=rows.merge(metadata[keys+['side','PassOrFail','TimeStamp','EQUIP_CD']],on=keys,how='left',validate='one_to_one',suffixes=('','_metadata'),sort=False)
    if not result.side.isin(['LH','RH']).all(): raise ValueError('Missing/invalid side')
    if not np.array_equal(result.PassOrFail,result.PassOrFail_metadata): raise ValueError('Metadata label mismatch')
    if not result[keys].equals(rows[keys]): raise ValueError('Row order changed')
    return result


def main(base,archive,out):
    if out.exists(): raise FileExistsError(out)
    cfg=json.loads((base/'config.json').read_text())
    splits=json.loads((base/'splits.json').read_text())
    old_manifest=json.loads((base/'manifest.json').read_text())
    features=json.loads((base/'preflight.json').read_text())['retained_features']
    protected={str(p):sha(p) for p in base.rglob('*') if p.is_file()}
    raw_paths=[PROJECT/'dataset'/f'moldset_{kind}_{g}.csv' for kind in ['labeled','unlabeled'] for g in ['cn7','rg3']]
    protected.update({str(p):sha(p) for p in raw_paths})
    metadata=[]; raw_frames=[]
    with zipfile.ZipFile(archive) as z:
        intermediate=pd.read_csv(z.open('dataset/moldset_labeled.csv'))
        for group in ['cn7','rg3']:
            name=f'moldset_labeled_{group}.csv'
            p=PROJECT/'dataset'/name
            if sha(p)!=old_manifest['input_sha256'][name]: raise ValueError('Raw changed')
            df=pd.read_csv(p)
            source=intermediate[intermediate.PART_NAME.str.startswith(group.upper())].iloc[:len(df)].copy().reset_index(drop=True)
            cols=[c for c in df if c not in ['Unnamed: 0','PassOrFail']]
            np.testing.assert_allclose(StandardScaler().fit_transform(source[cols]),df[cols],rtol=0,atol=1e-10)
            np.testing.assert_array_equal(source.PassOrFail,df.PassOrFail)
            source['side']=source.PART_NAME.str.extract(r'(LH|RH)$',expand=False)
            source['source_file']=name;source['row_position']=np.arange(len(source))
            metadata.append(source[['source_file','row_position','side','PassOrFail','TimeStamp','EQUIP_CD']])
            df['source_file']=name;df['row_position']=np.arange(len(df))
            raw_frames.append(df)
    meta=pd.concat(metadata,ignore_index=True)
    raw=pd.concat(raw_frames,ignore_index=True)
    prepared={}
    for scenario in cfg['scenarios']:
        rows=pd.read_csv(base/f'{scenario}_rows.csv')
        joined=join_metadata(rows,meta)
        # Explicit row-key alignment; only process inputs and known product/side identities enter model.
        frame=rows[['source_file','row_position']].merge(raw,on=['source_file','row_position'],how='left',validate='one_to_one',sort=False)
        np.testing.assert_array_equal(frame.PassOrFail,rows.PassOrFail)
        x=frame[features].copy()
        if scenario=='pooled': x['product_is_rg3']=rows.product_group.eq('rg3').astype(int)
        augmented=x.assign(part_is_rh=joined.side.eq('RH').astype(int))
        prepared[scenario]=(rows,joined,x,augmented,rows.PassOrFail.to_numpy())
    side=PROJECT/'artifacts/side_feature_v1'
    protected.update({str(p):sha(p) for p in side.rglob('*') if p.is_file()})
    old_protocol=json.loads((side/'protocol.json').read_text())
    assert sha(archive)==old_protocol['source_archive_sha256']
    assert cfg==json.loads((side/'config.json').read_text())
    for path,value in old_protocol['raw_and_baseline_sha256'].items(): assert sha(Path(path))==value
    out.mkdir(parents=True)
    names=['svm_rbf','svm_rbf_balanced','random_forest','random_forest_balanced','gaussian_nb','dnn_mlp']
    protocol={'models':names,'inputs':'same process + product (pooled only) + side as side_feature_v1','positive_label':1,'seed':42,'svm':{'C':1.0,'gamma':'scale','kernel':'rbf','probability':False,'score_transform':'expit(decision_function), ranking score NOT calibrated probability; 0.5 means margin zero'},'random_forest':{'n_estimators':200,'max_depth':8,'min_samples_leaf':3,'max_features':'sqrt','n_jobs':4},'gaussian_nb':{'var_smoothing':1e-9},'dnn':{'implementation':'sklearn MLPClassifier','hidden_layer_sizes':[64,32],'activation':'relu','solver':'adam','alpha':0.0001,'batch_size':64,'learning_rate_init':0.001,'max_iter':200,'early_stopping':False,'tol':0.0,'n_iter_no_change':201,'description':'two hidden fully-connected layers; fixed 200 epochs; no pseudo labels or random validation split'},'preprocessing':'VarianceThreshold and StandardScaler fitted inside each training fold (except RF: variance only)','threshold':'F1 maximum on existing inner group OOF only','ranking':'same tie keys and fold budgets','protected_files':protected,'code_sha256':sha(Path(__file__)),'baseline_helper_sha256':sha(PROJECT/'scripts/train_baseline.py'),'independent_test':False,'scope':'supervised baseline candidates, not guidebook reproduction or tuned winners'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    shutil.copyfile(__file__,out/'compare_extra_models_snapshot.py')
    old=pd.read_csv(side/'fold_metrics.csv');old=old[old.variant.eq('with_side')].copy()
    old['experiment']='existing_side_feature'
    old_oof=pd.read_csv(side/'oof_predictions.csv')
    scores=[];predictions=[];fit_logs=[];verified=0
    for split in splits:
        scenario=split['scenario'];rows,joined,x,aug,y=prepared[scenario]
        tr,va=np.array(split['train_indices']),np.array(split['validation_indices'])
        tg=joined.EQUIP_CD.astype(str)+'|'+joined.TimeStamp.astype(str)
        def disjoint(a,b):
            assert not set(rows.feature_group.iloc[a]) & set(rows.feature_group.iloc[b])
            assert not set(tg.iloc[a]) & set(tg.iloc[b])
        disjoint(tr,va)
        inner=[]
        for fold in split['inner_splits_relative_to_train']:
            it,iv=np.array(fold['train']),np.array(fold['validation']);disjoint(tr[it],tr[iv]);inner.append((it,iv))
        if split['seed']==42 and split['fold']==0:
            m=make_model('catboost_balanced',cfg,y[tr]);m.fit(aug.iloc[tr],y[tr])
            reference=old_oof[(old_oof.scenario==scenario)&(old_oof.seed==42)&(old_oof.fold==0)&(old_oof.model=='catboost_balanced')].set_index(['source_file','row_position'])
            reference=reference.loc[pd.MultiIndex.from_frame(rows.iloc[va][['source_file','row_position']])]
            np.testing.assert_allclose(probability(m,aug.iloc[va]),reference.p_class_1,rtol=0,atol=1e-12);verified+=1
        for name in names:
            ip=np.full(len(tr),np.nan)
            for inner_index,(it,iv) in enumerate(inner):
                m=extra_model(name,protocol)
                fit_logged(m,aug.iloc[tr[it]],y[tr[it]],fit_logs,scenario,split,name,'inner_'+str(inner_index))
                ip[iv]=probability(m,aug.iloc[tr[iv]])
            assert np.isfinite(ip).all()
            cutoff=select_threshold(y[tr],ip)
            m=extra_model(name,protocol)
            fit_logged(m,aug.iloc[tr],y[tr],fit_logs,scenario,split,name,'outer')
            p=probability(m,aug.iloc[va])
            common=dict(scenario=scenario,seed=split['seed'],fold=split['fold'],model=name,variant='with_side',experiment='extra_models',selected_threshold=cutoff)
            for policy,t in [('fixed_0.5',0.5),('inner_f1',cutoff)]:
                values=metrics(y[va],p,p>=t,rows.tie_key.to_numpy()[va],cfg['ranking_budgets'])
                if name.startswith('svm'):values['brier']=np.nan;values['log_loss']=np.nan
                scores.append({**common,'policy':policy,**values})
            pred=rows.iloc[va][['source_file','row_position','product_group','PassOrFail','feature_group','tie_key']].copy()
            predictions.append(pred.assign(**common,score=p,prediction_inner_f1=(p>=cutoff).astype(int)))
        print('Completed',split['key'],flush=True)
    combined=pd.concat([old,pd.DataFrame(scores)],ignore_index=True)
    combined['accuracy']=(combined.tp+combined.tn)/combined.rows
    combined.to_csv(out/'fold_metrics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(out/'oof_predictions.csv',index=False)
    pd.DataFrame(fit_logs).to_csv(out/'fit_diagnostics.csv',index=False)
    keys=['accuracy','ap','roc_auc','f1','precision','recall','recall_at_10pct']
    summary=combined.groupby(['scenario','model','policy'])[keys].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns]
    summary=summary.reset_index();summary.to_csv(out/'summary.csv',index=False)
    table=summary[summary.policy.eq('inner_f1')].sort_values(['scenario','ap_mean'],ascending=[True,False]);table.to_csv(out/'comparison.csv',index=False)
    lines=['# 추가 모델 비교: 좌우 정보 포함','', '기존 11설정과 SVM 2설정·Random Forest 2설정·GaussianNB·DNN 1설정씩을 합쳐 총 17설정이다. CN7/RG3/pooled에서 기존 3 seeds×3 folds와 내부 그룹 분할을 그대로 사용했다. 새 162개 외부 모델을 학습했다.', '', 'DNN은 sklearn MLPClassifier로 구현한 은닉층 64·32의 완전연결 신경망이다. 고정 200 epoch로 학습하며 random early stopping 검증을 만들지 않는다. TensorFlow/Keras 가이드북 모델이나 미라벨 의사라벨링을 재현한 것이 아니다. 수렴 진단은 fit_diagnostics.csv에 저장한다. 고정 epoch 종료 경고는 숨겨진 실패가 아니라 해당 학습 예산의 한계다.', '', 'SVM은 내부 무작위 확률 보정을 사용하지 않는다. decision_function에 sigmoid를 적용한 순위 점수로 AP/ROC-AUC/Recall@10%를 평가하며, 0.5는 margin=0이다. 이 점수는 보정된 불량 확률이 아니므로 SVM의 Brier/logloss는 보고하지 않는다. 임계값은 모든 모델에서 기존 내부 OOF F1 규칙으로 정한다.', '', '표는 9개 외부 폴드 평균. F1/Accuracy는 inner_f1 정책이다. 표준편차는 summary.csv에서 확인한다.', '', '|데이터|모델|AP|F1|Recall@10%|Accuracy|','|---|---|---:|---:|---:|---:|']
    for r in table.itertuples():lines.append(f'|{r.scenario}|{r.model}|{r.ap_mean:.4f}|{r.f1_mean:.4f}|{r.recall_at_10pct_mean:.4f}|{r.accuracy_mean:.4f}|')
    lines+=['','이 결과는 제한된 기본 설정의 개발 비교다. 같은 데이터로 EDA와 여러 모델 선택을 반복했으므로 가장 높은 평균을 독립 최종 성능으로 주장하지 않는다. 가이드북의 평가 불량수·분할·준지도 방식과 다르다. 데이터 표준화 및 CN7 시간대 집중의 한계도 유지된다. 기존 baseline·side_feature 파일은 해시로 변경 없음을 확인한다.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    for path,value in protected.items(): assert sha(Path(path))==value,path
    assert len(scores)==324
    (out/'verification.json').write_text(json.dumps({'status':'passed','new_outer_fits':162,'new_inner_fits':486,'reference_side_model_refits':verified,'raw_baseline_side_unchanged':True,'process_and_time_groups_disjoint':True,'independent_test':False},indent=2))

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from scipy.special import expit
import warnings
from sklearn.exceptions import ConvergenceWarning

class MarginSVC(ClassifierMixin,BaseEstimator):
    def __init__(self,class_weight=None):self.class_weight=class_weight
    def fit(self,x,y):
        self.model_=SVC(C=1.0,kernel='rbf',gamma='scale',class_weight=self.class_weight,probability=False,random_state=42)
        self.model_.fit(x,y);self.classes_=self.model_.classes_;return self
    def predict_proba(self,x):
        p=expit(self.model_.decision_function(x));return np.column_stack([1-p,p])
    def predict(self,x):return self.model_.predict(x)

def extra_model(name,protocol):
    steps=[('variance',VarianceThreshold())]
    if not name.startswith('random_forest'):steps.append(('scale',StandardScaler()))
    if name.startswith('svm'):m=MarginSVC('balanced' if name.endswith('balanced') else None)
    elif name.startswith('random_forest'):m=RandomForestClassifier(**protocol['random_forest'],random_state=42,class_weight='balanced' if name.endswith('balanced') else None)
    elif name=='gaussian_nb':m=GaussianNB(var_smoothing=1e-9)
    elif name=='dnn_mlp':m=MLPClassifier(hidden_layer_sizes=(64,32),activation='relu',solver='adam',alpha=.0001,batch_size=64,learning_rate_init=.001,max_iter=200,early_stopping=False,tol=0.,n_iter_no_change=201,random_state=42)
    else:raise ValueError(name)
    return Pipeline(steps+[('model',m)])

def fit_logged(model,x,y,logs,scenario,split,name,stage):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning);model.fit(x,y)
    core=model.named_steps['model']
    logs.append(dict(scenario=scenario,seed=split['seed'],fold=split['fold'],model=name,stage=stage,iterations=getattr(core,'n_iter_',np.nan),loss=getattr(core,'loss_',np.nan),warnings=' | '.join(str(w.message) for w in caught)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline-dir',type=Path,default=PROJECT/'artifacts/baseline_v2');p.add_argument('--archive',type=Path,default=PROJECT/'04. Dataset_Molding.zip');p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=4):main(a.baseline_dir,a.archive,a.output_dir)
