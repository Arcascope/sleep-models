"""Score saved frozen ensemble outputs with the actual public predictor's gap semantics."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from arcasleep_bidoze.adult import hybrid_calls
from arcasleep.ggir import evaluate_epoch_grid
from boat_evidence.score import DIVISOR, metrics
from boat_observation.sampling import summarize


def main(a):
    meta=pd.read_csv(a.data/'inventory.csv').set_index('name').to_dict('index');audit=[]
    for marker in sorted((a.out/'nights').glob('*/complete.json')):
        folder=marker.parent;r=meta[folder.name]
        with np.load(a.data/'nights'/f'{folder.name}.npz') as src:ggir=evaluate_epoch_grid(src['anglez'][:int(src['n'])])
        with np.load(folder/'probabilities.npz') as d:
            truth=d['truth'];still=d['still'];common=d['common_missing'];rows=[]
            for variant in ('native','decimate3','lowpass_native','lowpass_decimate3'):
                if variant not in d:continue
                p=d[variant]
                with np.load(folder/f'{variant}_axis_psd.npz') as v:missing=v['gap']
                # Predictor reports stage-conditional probabilities; the learned gap head
                # is telemetry, while only absent input epochs become gap calls.
                raw=hybrid_calls(p[:,:4],p[:,:4],DIVISOR);raw[missing]=4
                filtered=raw.copy();filtered[(raw>0)&(raw<4)&~ggir.is_sib&~ggir.is_spt]=0
                for mode,call in [('raw',raw),('ggir',filtered)]:
                    rows.append(dict(name=folder.name,cohort=r['cohort'],participant=r['participant'],group=r['group'],variant=variant,mode=mode)|metrics(truth,p,call,still,common,True))
                item=dict(name=folder.name,variant=variant,learned_gap_argmax_covered=int(((p.argmax(1)==4)&~common).sum()))
                if variant=='native':
                    stem=folder.name.split('_subject_',1)[1]
                    ref=a.baseline/'predictions/bidoze-adult'/f'{stem}.csv'
                    b=pd.read_csv(ref);k=np.rint(b.start_s/30).astype(int).to_numpy();keep=(k>=0)&(k<len(p));k=k[keep]
                    expected=np.array([{'wake':0,'light':1,'deep':2,'rem':3,'gap':4}.get(x,-1) for x in b.stage_raw.to_numpy()[keep]])
                    good=~common[k]&(expected>=0)
                    item.update(reference_covered_epochs=int(good.sum()),raw_call_differences=int((raw[k][good]!=expected[good]).sum()))
                audit.append(item)
        pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
    summarize(a.out)
    pd.DataFrame(audit).to_csv(a.out/'analysis/release-decoding-audit.csv',index=False)
    (a.out/'analysis/release-decoding-protocol.json').write_text(json.dumps(dict(ensemble='mean five-way probabilities; stage-conditional decoding',gap='physical missing input only; learned gap head is not a gap call',reference='existing boat-baseline released-ensemble raw calls; report numerical differences',aggregation='recording within participant then participant macro',coverage='all-variant common coverage',status='complete-cohort report only when all per-night completion markers exist'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('out',type=Path);p.add_argument('baseline',type=Path)
    main(p.parse_args())
