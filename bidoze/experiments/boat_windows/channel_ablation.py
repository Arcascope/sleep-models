"""Matched inference-only removal of one saved pooling-CNN channel."""
import argparse,csv,hashlib,json,time
from pathlib import Path
from flax.core import unfreeze,freeze
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr
from arcasleep_bidoze.model import BiDozeNetModel,BiDozeNetConfig
from arcasleep_bidoze.adult import hybrid_calls
from arcasleep.ggir import evaluate_epoch_grid
from pisces_lite.proc.constants import SPECTROGRAM_PADDING_VALUE as PAD
from boat_axes.score_training import metrics,DIVISOR,METRICS
from boat_windows.prepare import LAYOUTS

def main(a):
    a.out.mkdir(parents=True,exist_ok=True); (a.out/'predictions').mkdir(exist_ok=True)
    folder=a.run/'training/nonoverlap_cnn/D/seed0'; plan=json.loads((folder/'split.json').read_text())
    meta={r['name']:r for r in csv.DictReader((a.run/'setup/data/inventory.csv').open())}
    basis=np.load(a.run/'hypnograms/cnn-inspection/basis.npz'); ch=a.channel; mean=float(basis['mean'][128+ch])
    model=BiDozeNetModel(BiDozeNetConfig.from_dict(json.loads((folder/'bundle/config.json').read_text())))
    model.load_weights_safetensors(folder/'bundle/model.safetensors'); original=model.params
    variants={'baseline':original}
    for label,value in [('zero',0.),('fit_mean',mean)]:
        assert value>=0
        p=unfreeze(original); k=p['frame_pool']['conv']['kernel']; b=p['frame_pool']['conv']['bias']
        p['frame_pool']['conv']['kernel']=k.at[:,:,ch].set(0.)
        replacement=0. if value==0 else brentq(lambda v:v*ndtr(v)-value,0,max(1,value*2))
        p['frame_pool']['conv']['bias']=b.at[ch].set(replacement)
        # All parameters outside this one filter remain identical.
        for root in original:
            if root!='frame_pool':
                assert all(np.array_equal(x,y) for x,y in zip(jax.tree.leaves(original[root]),jax.tree.leaves(p[root])))
        assert np.array_equal(np.delete(np.asarray(k),ch,axis=2),np.delete(np.asarray(p['frame_pool']['conv']['kernel']),ch,axis=2))
        variants[label]=freeze(p)
    @jax.jit
    def predict(p,x):return jax.nn.softmax(model.module.apply({'params':p},x,train=False),axis=-1)
    names=[(role,n) for role in ('eval','test','guard') for n in plan[role]]
    if a.limit:names=names[:a.limit]
    protocol=dict(channel=ch,display_row=int(np.flatnonzero(basis['orders'][2]==ch)[0]),model='nonoverlap_cnn/D/seed0',fit_mean=mean,intervention='zero one complete pooling filter and set bias to inverse-GELU of replacement; exactly constant post-GELU output',variants=['baseline','zero','fit_mean'],baseline_check='fresh inference versus saved probabilities, absolute tolerance 0.002; paired comparison uses fresh baseline from same compiled function',aggregation='night means within participant then participant macro; paired differences',selection='channel selected by visual inspection of eval recording; exploratory, not an independent confirmation',devices=[str(d) for d in jax.devices()],checkpoint_sha256=hashlib.sha256((folder/'bundle/model.safetensors').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)); print(protocol,flush=True)
    rows=[]; shifts=[]; audits=[];start=time.monotonic()
    for i,(role,n) in enumerate(names,1):
        with np.load(a.run/'setup/data/nights'/f'{n}.npz') as d:
            ne=int(d['n']);frames=d['frames'].reshape(ne,15,60,2).copy(); frames[:,[j for j in range(15) if j not in LAYOUTS['nonoverlap']]]=PAD
            x=np.full((3000,15,60,2),PAD,dtype=np.float32);x[:ne]=frames;x=jnp.asarray(x.reshape(1,45000,60,2))
            gapfrac=d['x_nonoverlap'][:ne,-1,0].copy()
        with np.load(folder/'predictions'/f'{n}.npz') as d:
            y=d['y'];missing=d['missing'];still=d['still'];anglez=d['anglez'];saved=d['probabilities']
        ps={label:np.asarray(predict(p,x))[0,:ne] for label,p in variants.items()}
        delta=float(np.max(np.abs(ps['baseline']-saved)));assert np.allclose(ps['baseline'],saved,atol=0.002,rtol=0),(n,delta)
        audits.append(dict(name=n,baseline_max_abs_difference=delta))
        np.savez_compressed(a.out/'predictions'/f'{n}.npz',**ps,y=y,missing=missing,gap_fraction=gapfrac)
        ggir=evaluate_epoch_grid(anglez);calls={}
        for label,p in ps.items():
            raw=hybrid_calls(p[:,:4],p[:,:4],DIVISOR);raw[missing]=4
            filtered=raw.copy();filtered[(raw>0)&(raw<4)&~ggir.is_sib&~ggir.is_spt]=0;calls[label]=(raw,filtered)
            for mode,c in [('raw',raw),('ggir',filtered)]:
                rows.append(dict(name=n,role=role,cohort=meta[n]['cohort'],participant=meta[n]['participant'],variant=label,mode=mode)|metrics(y,p,c,still,missing,True))
        near=np.convolve((gapfrac>0).astype(int),np.ones(5,dtype=int),mode='same')>0
        masks={'all_available':~missing,'fully_covered':(~missing)&(gapfrac==0),'partially_covered':(~missing)&(gapfrac>0),'near_missing_2epochs':(~missing)&near,'away_from_missing':(~missing)&~near,'unavailable':missing}
        for label in ('zero','fit_mean'):
            diff=ps[label]-ps['baseline']
            for stratum,take in masks.items():
                if not take.any():continue
                shifts.append(dict(name=n,role=role,cohort=meta[n]['cohort'],participant=meta[n]['participant'],variant=label,stratum=stratum,epochs=int(take.sum()),mean_abs_probability_change=float(np.abs(diff[take]).mean()),max_abs_probability_change=float(np.abs(diff[take]).max()),argmax_flip_fraction=float((ps[label][take].argmax(-1)!=ps['baseline'][take].argmax(-1)).mean()),release_flip_fraction=float((calls[label][1][take]!=calls['baseline'][1][take]).mean()),wake_probability_change=float(diff[take,0].mean())))
        pd.DataFrame(audits).to_csv(a.out/'baseline_audit.csv',index=False)
        print(i,len(names),n,'baseline_delta',delta,'seconds',round(time.monotonic()-start,1),flush=True)
    nights=pd.DataFrame(rows);nights.to_csv(a.out/'nights.csv',index=False)
    keys=['role','cohort','participant','variant','mode'];subjects=nights.groupby(keys)[METRICS].mean().reset_index();subjects.to_csv(a.out/'subjects.csv',index=False)
    subjects.groupby(['role','cohort','variant','mode'])[METRICS].mean().to_csv(a.out/'summary.csv')
    paired=[]
    for (role,cohort,mode),g in subjects.groupby(['role','cohort','mode']):
        for metric in METRICS:
            wide=g.pivot(index='participant',columns='variant',values=metric)
            for label in ('zero','fit_mean'):
                pair=wide[['baseline',label]].dropna();d=pair[label]-pair.baseline
                paired.append(dict(role=role,cohort=cohort,mode=mode,metric=metric,variant=label,participants=len(d),difference=float(d.mean())))
    pd.DataFrame(paired).to_csv(a.out/'paired.csv',index=False)
    sh=pd.DataFrame(shifts);sh.to_csv(a.out/'probability_shifts_nights.csv',index=False)
    sk=['role','cohort','participant','variant','stratum'];numeric=['mean_abs_probability_change','argmax_flip_fraction','release_flip_fraction','wake_probability_change'];ss=sh.groupby(sk)[numeric].mean().reset_index();ss.to_csv(a.out/'probability_shifts_subjects.csv',index=False);ss.groupby(['role','cohort','variant','stratum'])[numeric].mean().to_csv(a.out/'probability_shifts_summary.csv')
    (a.out/'complete.json').write_text(json.dumps(dict(recordings=len(names),seconds=time.monotonic()-start,max_baseline_difference=max(r['baseline_max_abs_difference'] for r in audits)),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('out',type=Path);p.add_argument('--channel',type=int,default=16);p.add_argument('--limit',type=int);main(p.parse_args())
