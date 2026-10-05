#!/usr/bin/env python3
"""One fixed ERP-template hypothesis on Cross development people only."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eeg_comp.data import load_cache
from eeg_comp.erp_template import ERPTemplateCovariance
from eeg_comp.validation import make_splits,summarize
from scripts.run_classical import cache_provenance,feature_cache,sha256_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--permutation',type=int)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    arrays,meta=load_cache('artifacts','cross_subject')
    directory=Path('artifacts/cross_subject')
    manifest,digest=cache_provenance(directory,arrays)
    erp,signature=feature_cache(directory,'erp_pre',arrays['X'],arrays['valid_mask'],arrays['phase_valid_mask'],digest)
    waveforms=np.asarray(erp).reshape(len(meta),8,50)
    del arrays
    args.output.mkdir(parents=True)
    config={'competition':'cross_subject','category':'ML','window':[0,2],
        'waveform':'Same binned baseline-standardized lowpass20Hz EEG as erp_pre; covariance removes per-channel temporal means.',
        'template':'Two source-class mean waveforms concatenated with each trial:24channelsx50bins',
        'covariance_shrinkage':.1,'C':.1,'source_cache':manifest,'feature_signature':signature,
        'permutation':args.permutation,'confirmation':'S019/S020 excluded from every fitting/query set',
        'information_budget':'Inductive per-trial query; templates/reference/scaler fit on sourcefold only',
        'code_sha256':{str(p):sha256_file(p) for p in [Path('eeg_comp/erp_template.py'),Path(__file__),Path('eeg_comp/classical.py')]}}
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    records=[];provenance=[];started=time.monotonic()
    for name,train,query in make_splits(meta,'cross_subject',set(range(5))):
        labels=meta.iloc[train].y.to_numpy().copy()
        if args.permutation is not None:
            rng=np.random.default_rng(args.permutation)
            source=meta.iloc[train].reset_index(drop=True)
            for _,g in source.groupby(['subject','run']):labels[g.index]=rng.permutation(labels[g.index])
        model=ERPTemplateCovariance().fit(waveforms[train],labels)
        p=model.predict_proba(waveforms[query])[:,1]
        frame=meta.iloc[query].copy();frame['fold']=name;frame['p_move']=p
        records.append(frame)
        provenance.append({'fold':name,'source_indices':train.tolist(),'query_indices':query.tolist()})
        print(name,'accuracy',float(np.mean((p>=.5)==frame.y)),flush=True)
    oof=pd.concat(records).sort_values('epoch_index')
    if len(oof)!=1575 or oof.epoch_index.duplicated().any():raise ValueError('Wrong outer coverage')
    oof.to_csv(args.output/'oof.csv',index=False)
    metrics=summarize(oof,meta);metrics['elapsed_seconds']=time.monotonic()-started
    (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    (args.output/'splits.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({k:metrics[k] for k in ['accuracy','target_weighted_accuracy','log_loss','mean_domain_auc','elapsed_seconds']}),flush=True)

if __name__=='__main__':main()
