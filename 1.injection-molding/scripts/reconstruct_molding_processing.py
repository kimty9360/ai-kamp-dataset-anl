"""Reconstruct supplied labeled transformations, not a recommended training pipeline.
Row selection is an empirically verified equivalence; original selection intent unknown.
"""
from pathlib import Path
import argparse, hashlib, json, zipfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
PART_NO={"CN7 W/S SIDE MLD'G LH":"86131AA000", "CN7 W/S SIDE MLD'G RH":"86141AA000",
         "RG3 MOLD'G W/SHLD, LH":"86131T1000", "RG3 MOLD'G W/SHLD, RH":"86141T1000"}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/processing_reconstruction_v1')
    out=parser.parse_args().output
    if out.exists(): raise FileExistsError('Choose a new output directory')
    archive=ROOT/'sources/04. Dataset_Molding.zip'; sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as z:
        raw=pd.read_csv(z.open('dataset/labeled_data.csv'))
        supplied=pd.read_csv(z.open('dataset/moldset_labeled.csv'))
        # This boundary matches the supplied files, not a documented quality filter.
        middle=raw.iloc[:2607].copy()
        middle.insert(4,'PART_NO',middle.PART_NAME.map(PART_NO))
        assert middle.PART_NO.notna().all()
        middle['PassOrFail']=middle.PassOrFail.map({'Y':0,'N':1})
        middle.insert(0,'Unnamed: 0',middle.index)
        middle=pd.concat([middle[middle.PART_NAME.str.startswith('CN7')],middle[middle.PART_NAME.str.startswith('RG3')]],ignore_index=True)
        pd.testing.assert_frame_equal(middle,supplied,check_exact=True)
        comparisons=[];scalers=[];examples=[];dropped={}
        for product in ['cn7','rg3']:
            final=pd.read_csv(z.open(f'dataset/moldset_labeled_{product}.csv'))
            subset=middle[middle.PART_NAME.str.startswith(product.upper())]
            chosen=subset.iloc[:1211] if product=='cn7' else subset
            cols=[c for c in final if c not in ['Unnamed: 0','PassOrFail']]
            scaler=StandardScaler().fit(chosen[cols])
            transformed=scaler.transform(chosen[cols])
            reconstructed=chosen[final.columns].copy().reset_index(drop=True)
            reconstructed[cols]=transformed
            np.testing.assert_allclose(reconstructed[cols],final[cols],rtol=0,atol=1e-10)
            pd.testing.assert_frame_equal(reconstructed[['Unnamed: 0','PassOrFail']],final[['Unnamed: 0','PassOrFail']],check_exact=True)
            dropped[product]=[c for c in middle if c not in final]
            comparisons.append({'product':product,'middle_rows':len(subset),'final_rows':len(final),'omitted_rows':len(subset)-len(chosen),'omitted_defects':int(subset.iloc[len(chosen):].PassOrFail.sum()),'process_columns':len(cols),'max_absolute_difference':float(np.abs(transformed-final[cols].to_numpy()).max())})
            for i,c in enumerate(cols):
                scalers.append({'product':product,'feature':c,'rows_used_for_fit':len(chosen),'mean':scaler.mean_[i],'population_std':float(np.sqrt(scaler.var_[i])),'scale_used':scaler.scale_[i]})
                examples.append({'product':product,'feature':c,'original_first_row_value':float(chosen[c].iloc[0]),'mean':scaler.mean_[i],'scale_used':scaler.scale_[i],'reconstructed_first_row_value':float(transformed[0,i]),'supplied_first_row_value':float(final[c].iloc[0])})
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==sha
    out.mkdir(parents=True)
    pd.DataFrame(comparisons).to_csv(out/'reconstruction_summary.csv',index=False)
    pd.DataFrame(scalers).to_csv(out/'scaler_parameters.csv',index=False)
    pd.DataFrame(examples).to_csv(out/'calculation_examples.csv',index=False)
    proof={'source_sha256':sha,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'middle_all_cells_exact_match':True,'middle_source_positions_zero_based':[[0,1210],[2393,2606],[1211,2392]],'middle_original_process_values_unchanged':True,'added_part_number_lookup':PART_NO,'removed_columns':dropped,'label_mapping':{'Y':0,'N':1},'standardization':'(x - mean) / population_std (ddof=0); constant column scale=1','float_tolerance':1e-10,'all_checks_passed':True,'source_unchanged':True,'limitations':['Equivalent reconstruction does not prove the original author used these exact code steps.','Reason for first 2607 raw rows and omitted 214 CN7 rows is undocumented.','PART_NO lookup is recovered from supplied middle file; not computable from process measurements.','Scaling all provided rows reproduces historical files; new predictive evaluation must fit preprocessing on training only.']}
    (out/'verification.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
    print(pd.DataFrame(comparisons).to_string(index=False));print('Middle: all 2607 x 47 cells exactly matched. Saved:',out)

if __name__=='__main__':main()
