"""Read-only lineage audit of the user-provided KAMP archive."""
import argparse, hashlib, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
PROJECT=Path(__file__).resolve().parents[1]
def digest(b): return hashlib.sha256(b).hexdigest()
def main(archive,out):
    if out.exists(): raise FileExistsError(out)
    with zipfile.ZipFile(archive) as z:
        raw=pd.read_csv(z.open('dataset/labeled_data.csv'))
        middle=pd.read_csv(z.open('dataset/moldset_labeled.csv'))
        lookup=raw[['_id','PassOrFail']].drop_duplicates()
        assert lookup._id.is_unique and middle._id.is_unique
        joined=middle.merge(lookup,on='_id',suffixes=('_numeric','_original'),validate='one_to_one')
        assert len(joined)==len(middle)
        assert np.array_equal(joined.PassOrFail_numeric,joined.PassOrFail_original.map({'Y':0,'N':1}))
        assert middle.loc[middle.PassOrFail.eq(1),'Reason'].notna().all()
        results=[]; details=[]; distributions=[]
        files={}
        for g in ['cn7','rg3']:
            for kind in ['labeled','unlabeled']:
                name=f'moldset_{kind}_{g}.csv'
                files[name]=digest(z.read('dataset/'+name))
                assert files[name]==digest((PROJECT/'dataset'/name).read_bytes())
            final=pd.read_csv(z.open(f'dataset/moldset_labeled_{g}.csv'))
            subset=middle[middle.PART_NAME.str.startswith(g.upper())].copy()
            source=subset.iloc[:len(final)].copy().reset_index(drop=True)
            features=[c for c in final if c not in ['Unnamed: 0','PassOrFail']]
            reconstructed=StandardScaler().fit_transform(source[features])
            error=float(np.max(abs(reconstructed-final[features].to_numpy())))
            np.testing.assert_allclose(reconstructed,final[features],rtol=0,atol=1e-10)
            np.testing.assert_array_equal(source.PassOrFail,final.PassOrFail)
            source['row_position']=np.arange(len(source))
            source['product_group']=g
            source['side']=source.PART_NAME.str.extract(r'(LH|RH)$',expand=False)
            assert source.side.notna().all()
            source['feature_group']=final.groupby(features,sort=True).ngroup()
            grouped=source.groupby('feature_group')
            conflict=grouped.PassOrFail.nunique().eq(2)
            ids=conflict[conflict].index
            conflict_rows=source[source.feature_group.isin(ids)]
            cs=conflict_rows.groupby('feature_group')
            same_record_conditions=(cs.TimeStamp.nunique().eq(1)&cs.EQUIP_CD.nunique().eq(1)&cs.side.nunique().eq(2)&cs.size().eq(2))
            for (side,label),d in source.groupby(['side','PassOrFail']):
                distributions.append(dict(product_group=g,side=side,label=int(label),rows=len(d)))
            positive=source[source.PassOrFail.eq(1)]
            omitted=subset.iloc[len(final):]
            results.append(dict(product_group=g,intermediate_rows=len(subset),provided_rows=len(final),matching_rule='first N product rows, original order; independent StandardScaler per product',max_abs_error=error,omitted_rows=len(omitted),omitted_class1=int(omitted.PassOrFail.sum()),conflict_groups=len(ids),conflict_pairs_same_timestamp_equipment_opposite_side=int(same_record_conditions.sum()),date_min=source.TimeStamp.min(),date_max=source.TimeStamp.max(),class1_time_min=positive.TimeStamp.min(),class1_time_max=positive.TimeStamp.max(),timestamps_monotonic=bool(pd.to_datetime(source.TimeStamp).is_monotonic_increasing)))
            columns=['product_group','row_position','_id','TimeStamp','PART_FACT_SERIAL','PART_NO','PART_NAME','EQUIP_CD','side','PassOrFail','Reason','feature_group']
            details.append(source[columns].assign(label_conflict=source.feature_group.isin(ids)))
    out.mkdir(parents=True);(out/'rows').mkdir()
    pd.DataFrame(results).to_csv(out/'lineage_summary.csv',index=False)
    pd.DataFrame(distributions).to_csv(out/'side_label_counts.csv',index=False)
    pd.concat(details).to_csv(out/'rows/matched_metadata.csv',index=False)
    provenance={'archive_sha256':digest(archive.read_bytes()),'code_sha256':digest(Path(__file__).read_bytes()),'raw_rows':len(raw),'raw_unique_ids':int(raw._id.nunique()),'intermediate_rows':len(middle),'id_matches':len(joined),'label_mapping':{'Y':0,'N':1},'official_processed_hashes':files,'tolerance':1e-10,'all_checks_passed':True,'limitations':['No evidence why CN7 last 214 rows were omitted.','Same timestamp/equipment supports paired records, but is not a proven physical shot ID.','Metadata mapping is verified by ordered numerical reconstruction, not IDs retained in processed CSV.','Reason is post-inspection target information and must not be a predictive feature.']}
    (out/'verification.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    lines=['# 原資料 연결 및 좌우 부품 진단'.replace('原資料','원자료'),'','## 라벨 및 파일 확인','',f'ZIP의 현재 가공 CSV 네 개는 로컬 대회 파일과 바이트 단위로 일치한다. labeled_data.csv의 중복 ID는 라벨이 모두 일치하며 ID·라벨 조합을 중복 제거한 뒤 중간본 {len(joined)}행을 ID로 연결했다. Y→0, N→1이 전 행에서 성립하고 라벨 1에는 모두 불량 사유가 존재한다. 가이드와 합쳐 현재 타깃은 0=양품, 1=불량으로 해석한다.','','## 가공본 재현','','|제품군|중간본 행|현재 행|제외 행(라벨1)|최대 표준화 오차|충돌 그룹|동시각·동일설비·반대쪽 2행인 충돌 그룹|','|---|---:|---:|---:|---:|---:|---:|']
    for r in results:lines.append(f"|{r['product_group']}|{r['intermediate_rows']}|{r['provided_rows']}|{r['omitted_rows']} ({r['omitted_class1']})|{r['max_abs_error']:.2e}|{r['conflict_groups']}|{r['conflict_pairs_same_timestamp_equipment_opposite_side']}|")
    lines+=['','CN7은 중간본의 제품별 앞 1211행, RG3는 전체 1182행을 원래 순서대로 선택하고 각 파일에서 StandardScaler를 fit하면 모든 24개 공정값과 라벨이 재현된다. CN7 뒤 214행(라벨 1 10행)의 제외 이유는 파일만으로 확인되지 않는다.','','## 좌우별 라벨','','|제품군|좌우|라벨|행|','|---|---|---:|---:|']
    for r in distributions:lines.append(f"|{r['product_group']}|{r['side']}|{r['label']}|{r['rows']}|")
    lines+=['','## 다음 실험 원칙','','- 공정값만으로는 같아 보인 충돌 그룹을 좌우 정보를 통해 구별할 근거가 생겼다. 단순 라벨 오류로 삭제하지 않는다.','- 좌우 변수를 추가해도 기존 원공정 그룹을 유지해 한 쌍이 학습·평가에 나뉘지 않게 한다.','- TimeStamp·설비를 이용한 보수적 생산 그룹도 점검한다. 충돌 쌍의 생산일련번호는 서로 달라 이것을 동일 shot ID로 사용하지 않는다. 같은 시각은 실제 shot ID의 대체 가정임을 명시한다.','- Reason은 검사 후 정보로 입력 금지. 원자료 전체를 기존 평가에 무심코 합치지 않는다.','- 물리 단위 입력을 사용하려면 학습 폴드에서만 전처리를 fit하는 새 실험으로 분리한다. 기존 가공 데이터의 검증 성능과 구분한다.','- 라벨1이 RH에만 나타나는 제품군에서도 side가 실제 원인이라는 뜻은 아니다. 다른 기간에서도 유지되는지 평가가 필요하다.','','행별 연결 정보는 rows/에 로컬 저장하고 공개 Git에서 제외한다. 이번 진단은 모델을 학습하거나 원본을 수정하지 않았다.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(results,ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,default=PROJECT/'04. Dataset_Molding.zip');p.add_argument('--output-dir',type=Path,default=PROJECT/'artifacts/lineage_v1');a=p.parse_args();main(a.archive,a.output_dir)
