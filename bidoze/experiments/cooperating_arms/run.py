"""Train one cooperating-arm fit on an existing mixed-cohort draw, then save all arm outputs.

All paths are explicit; no source cache is modified. Run report.py after each completed fit.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
from arcasleep.data.cache import discover_cached_subjects, load_cached_features, participant_id
from arcasleep.data.batching import assemble_batch
from arcasleep.training.balanced import Patience, split_train_eval
from arcasleep_bidoze.model import BiDozeNetConfig
from arcasleep_bidoze.adult import stage_counts, adult_stage_divisor
from model import ArmSpec, CooperatingModel

CACHE = '12Hz_log_PSD_aligned'
TRAIN = ['weaverDraw', 'trainSleepAccel', 'trainClinical']
TEST = ['testSleepAccel', 'testClinical']


def timestamp(): return datetime.now().astimezone().isoformat(timespec='seconds')


def configuration(arm):
    kind = arm.rstrip('0123456789')
    n = int(arm[len(kind):])
    widths = {1: 256, 3: 124, 5: 88, 7: 68}
    shared_widths = {3: 140, 5: 108, 7: 88}
    if kind not in ('final', 'exchange', 'shared', 'control'):
        raise ValueError(arm)
    s = ArmSpec(members=n, width=shared_widths[n] if kind == 'shared' else widths[n],
                shared_cnn=kind == 'shared', exchange_after=(2,) if kind == 'exchange' else ())
    cfg = BiDozeNetConfig(num_classes=4, score_gap=True, d_model=s.width,
        n_heads=8 if n == 1 else 4, d_ff=2*s.width, n_transformer_layers=4,
        conv_kernels=(3, 7, 31, 63), dropout=.2, attn_dropout=0, gap_from_row=True)
    return cfg, s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--source-run', type=Path, required=True)
    ap.add_argument('--labels-root', type=Path, required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--draw', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-seconds', type=int, default=7200)
    ap.add_argument('--epochs', type=int, default=3000)
    ap.add_argument('--prepare-only', action='store_true')
    a = ap.parse_args()
    out = a.run / a.arm / f'draw{a.draw}_seed{a.seed}'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'complete.json').exists() and not json.loads((out/'complete.json').read_text()).get('time_capped',False):
        raise RuntimeError(f'already complete: {out}')
    cfg, spec = configuration(a.arm)
    # Tiny initialization counts real tensors without allocating long attention surfaces.
    model = CooperatingModel(cfg, spec)
    model._ensure_built((8, 61, 2), seed=a.seed)
    counts = {k: sum(int(x.size) for x in jax.tree.leaves(v)) for k,v in model.params.items()}
    model.input_shape = (3000, 61, 2)
    config = dict(arm=a.arm, draw=a.draw, seed=a.seed, config=cfg.to_dict(), spec=asdict(spec),
        parameters=sum(counts.values()), parameter_groups=counts,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        started=timestamp(), source_run=str(a.source_run.resolve()),
        objective='balanced_cross_entropy(mean(softmax(arm_logits)))',
        batch_size=2, max_epochs=a.epochs, max_train_seconds=a.max_seconds,
        patience=dict(lr_drop=700, early_stop=1750, unit='steps', smoothing_steps=350),
        initial_lr=.001, weight_decay=0, balance_beta=.999)
    (out/'config.json').write_text(json.dumps(config, indent=2))
    print(json.dumps({k:config[k] for k in ['arm','parameters','started','code_commit']}),flush=True)
    if a.prepare_only: return
    print('devices:',jax.devices(),flush=True)
    if not any(d.platform == 'gpu' for d in jax.devices()):
        raise RuntimeError('a local GPU is required for training')
    cache = a.source_run / 'setup/cache' / f'draw{a.draw}' / CACHE
    found = discover_cached_subjects(TRAIN+TEST, cache)
    refs = sorted(s.ref for ds in TRAIN for s in found[ds])
    fit, val = split_train_eval(refs,.3,0)
    tests = sorted(s.ref for ds in TEST for s in found[ds])
    groups = [set(participant_id(r.subject_id) for r in rs) for rs in [fit,val,tests]]
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2]), 'participant leakage'
    split = {name:[dict(dataset=r.dataset_name,subject_id=r.subject_id) for r in rs]
             for name,rs in [('fit',fit),('validation',val),('test',tests)]}
    split['counts'] = {name:dict(Counter(r.dataset_name for r in rs))
                       for name,rs in [('fit',fit),('validation',val),('test',tests)]}
    for name in ['fit','validation','test']:
        for cohort in (['testSleepAccel','testClinical'] if name=='test' else ['trainSleepAccel','trainClinical']):
            assert split['counts'][name].get(cohort,0)>0,(name,cohort)
    split['cache_sha256']={s.path.name:hashlib.sha256(s.path.read_bytes()).hexdigest()
                          for ds in TRAIN+TEST for s in found[ds]}
    (out/'split.json').write_text(json.dumps(split,indent=2))
    print('split:',split['counts'],flush=True)
    features = load_cached_features([s for ds in TRAIN for s in found[ds]])
    X,y=assemble_batch(features,fit,3000,3000,4)
    V,v=assemble_batch(features,val,3000,3000,4)
    del features
    # The divisor matches historical draw scoring: all nominal training nights, not test data.
    counts_train = np.array([stage_counts(np.load(s.path)['y']) for ds in TRAIN for s in found[ds]])
    adult_paths = sorted((a.labels_root/CACHE).glob('bidsleep_subject_*'))
    assert adult_paths,'BIDsleep reference missing'
    counts_adult=np.array([stage_counts(np.load(p)['y']) for p in adult_paths])
    divisor=adult_stage_divisor(counts_train,counts_adult,.999)
    np.save(out/'stage_divisor.npy',divisor)
    probe_rows=[next(i for i,r in enumerate(val) if r.dataset_name==ds)
                for ds in ['trainSleepAccel','trainClinical']]
    def callback(epoch,params,history):
        (out/'history.json').write_text(json.dumps(history,indent=2))
        if epoch in (1,5,10,25,50,100,200) or epoch % 250 == 0:
            probe = model.module.apply({'params':params},jnp.asarray(V[probe_rows]),jnp.ones(V[probe_rows].shape[:2]),
                                       return_outputs=True)['arm_probabilities']
            q=np.asarray(probe)
            np.savez_compressed(out/f'validation_probe_epoch{epoch:04d}.npz',probabilities=q,
                                labels=v[probe_rows],validation_refs=json.dumps([split['validation'][i] for i in probe_rows]))
    start=time.monotonic()
    history=model.fit_balanced(X,y,V,v,batch_size=2,initial_lr=.001,weight_decay=0,
        balance_beta=.999,patience=Patience(700,1750,'steps',350),max_epochs=a.epochs,
        fit_seed=a.seed,eval_batch_size=1,max_train_seconds=a.max_seconds,
        fit_state_path=out/'fit_state.pkl',epoch_callback=callback)
    elapsed=time.monotonic()-start
    (out/'history.json').write_text(json.dumps(history,indent=2))
    model.save_weights_to_directory(out/'weights')
    (out/'training_finished.json').write_text(json.dumps(dict(at=timestamp(),seconds=elapsed,
         time_capped=history.get('time_capped',False)),indent=2))
    print('saved selected weights; predicting test recordings',flush=True)
    del X,y,V,v
    jax.clear_caches()
    predict = jax.jit(lambda p,x: model.module.apply({'params':p},x,jnp.ones(x.shape[:2]),
                                                   return_outputs=True)['arm_probabilities'])
    pred=out/'predictions';pred.mkdir(exist_ok=True)
    from arcasleep.data.batching import recording_input,pad_pooled
    from arcasleep.data.bundle import CacheBundle
    for ds in TEST:
        for subject in found[ds]:
            bundle=CacheBundle.from_npz(subject.path)
            n=bundle.y.shape[1]
            if n>3000: raise ValueError('test recording longer than training window')
            inputs=recording_input(bundle.X[0],n,1)
            q=np.asarray(predict(model.params,jnp.asarray(pad_pooled(inputs,3000)[None])))[:,0,:n]
            labels=bundle.y_for_n_classes(4)[0]
            np.savez_compressed(pred/f'{ds}_{subject.subject_id}.npz',probabilities=q,labels=labels)
            print('predicted',ds,subject.subject_id,flush=True)
    done=dict(finished=timestamp(),train_seconds=elapsed,time_capped=history.get('time_capped',False),
              test_recordings=len(tests),parameters=config['parameters'])
    (out/'complete.json').write_text(json.dumps(done,indent=2))
    print('COMPLETE',json.dumps(done),flush=True)


if __name__=='__main__':main()
