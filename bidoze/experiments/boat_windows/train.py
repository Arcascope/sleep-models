"""Matched seed-controlled window containment and learned pooling screening."""
from __future__ import annotations
import argparse
import csv
import json
import os
import shutil
import subprocess
import time
import hashlib
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import optax
from arcasleep.artifacts.publish import load_bundle_model
from arcasleep_bidoze.model import BiDozeNetModel,BiDozeNetConfig
from arcasleep.training.balanced import balanced_cross_entropy, Patience, Plateau, selection_confusion
from arcasleep.evaluation.staging_metrics import cohen_kappa_from_confusion
from pisces_lite.proc.constants import SPECTROGRAM_PADDING_VALUE as PAD
from boat_windows.prepare import LAYOUTS


def main(a):
    a.out.mkdir(parents=True,exist_ok=True)
    plan=json.loads((a.data/'splits.json').read_text())[a.fold]
    metadata={r['name']:r for r in csv.DictReader((a.data/'inventory.csv').open())}
    if a.arm=='land':
        plan={**plan,'fit':[n for n in plan['fit'] if metadata[n]['role']=='fit'],
              'eval':[n for n in plan['eval'] if metadata[n]['role']=='eval'],
              'test':[n for n,r in metadata.items() if r['role']=='boat']}
    if a.smoke:
        plan={**plan,'fit':[plan['fit'][0],plan['fit'][-1]], 'eval':[plan['eval'][0],plan['eval'][-1]], 'test':plan['test'][:1], 'guard':plan['guard'][:1]}
    (a.out/'split.json').write_text(json.dumps(plan,indent=2))
    def view(r):
        if a.pooling=='mean':return r[f'x_{a.layout}']
        n=int(r['n']);frames=r['frames'].reshape(n,15,60,2).copy()
        excluded=[i for i in range(15) if i not in LAYOUTS[a.layout]]
        frames[:,excluded]=PAD
        out=np.full((3000,15,60,2),PAD,dtype=np.float32);out[:n]=frames
        return out.reshape(45000,60,2)
    def arrays(names):
        rows=[np.load(a.data/'nights'/f'{n}.npz') for n in names]
        x=np.stack([view(r) for r in rows]); y=np.stack([r['y'] for r in rows])
        for r in rows:r.close()
        return x,y
    xf,yf=arrays(plan['fit']); xe,ye=arrays(plan['eval'])
    bundle=a.data/'bundles'/f'seed{a.seed}'; release=load_bundle_model(bundle)
    cfg=json.loads((bundle/'config.json').read_text())
    cfg.update(frames_per_epoch=15 if a.pooling=='cnn' else 1,frame_pooling='conv' if a.pooling=='cnn' else 'mean',frame_pool_features=64,frame_pool_expected_frames=len(LAYOUTS[a.layout]) if a.pooling=='cnn' else None)
    model=BiDozeNetModel(BiDozeNetConfig.from_dict(cfg))
    model._ensure_built(tuple(xf.shape[1:]),seed=a.seed)
    params=model.params; module=model.module; nc=model.config.num_classes; oc=model.config.output_classes
    @jax.jit
    def logits(params,x):return module.apply({'params':params},x,train=False)
    def predict(p,x):return np.concatenate([np.asarray(jax.nn.softmax(logits(p,jnp.asarray(x[i:i+1])),axis=-1)) for i in range(len(x))])
    def kappa(p,y):return float(cohen_kappa_from_confusion(selection_confusion(y,p.argmax(-1),nc)))
    if a.arm=='head':
        labels=jax.tree_util.tree_map_with_path(lambda path,_: 'fit' if path[0].key in {'head_hidden','head_logits'} else 'freeze',params)
        tx=optax.multi_transform({'fit':optax.adam(a.lr),'freeze':optax.set_to_zero()},labels)
    else:tx=optax.scale_by_adam()
    state=tx.init(params)
    @jax.jit
    def step(p,s,x,y,key,learning_rate):
        mask=((y>=0)|(y==-2)).astype(jnp.float32); target=jnp.where(y<=-2,nc,y)
        if a.pooling=='cnn':mask=jnp.repeat(mask,15,axis=1)
        def loss(pp):
            lg=module.apply({'params':pp},x,mask,train=True,rngs={'dropout':key,'augment':key})
            return balanced_cross_entropy(lg,target,oc,0.999)
        val,grad=jax.value_and_grad(loss)(p); update,s=tx.update(grad,s,p)
        update=jax.tree.map(lambda v: -learning_rate*v,update)
        return optax.apply_updates(p,update),s,val
    reference=predict(params,xe); best=kappa(reference,ye); best_epoch=0
    dest=a.out/'bundle'; dest.mkdir(exist_ok=True)
    (dest/'config.json').write_text(json.dumps(cfg,indent=2))
    (dest/'input_recipe.json').write_text(json.dumps(dict(representation=a.layout+'_'+a.pooling,recipe=json.loads((a.data/'window-input-recipe.json').read_text()),status='custom cached research checkpoint; no packaged raw predictor'),indent=2))
    def save(p,epoch):
        model.params=p; model.model={'params':p}; model.save_weights_safetensors(dest/'model.safetensors.partial'); os.replace(dest/'model.safetensors.partial',dest/'model.safetensors')
        (dest/'arm.json').write_text(json.dumps(dict(arm=(a.layout+'_'+a.pooling),initialization="scratch",fold=a.fold,seed=a.seed,epoch=epoch,lr=a.lr)))
    save(params,0)
    initial_hash=hashlib.sha256(b''.join(np.asarray(v).tobytes() for v in jax.tree.leaves(params))).hexdigest()
    config=dict(initial_parameter_sha256=initial_hash,input_shape=list(xf.shape[1:]),arm=(a.layout+'_'+a.pooling),initialization="scratch",fold=a.fold,seed=a.seed,lr=a.lr,max_epochs=a.max_epochs,fit=len(xf),eval=len(xe),initial_eval_kappa=best,context=3000,layout=a.layout,pooling=a.pooling,selected_frames=len(LAYOUTS[a.layout]),devices=[str(d) for d in jax.devices()],pid=os.getpid(),code=os.environ.get('GIT_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (a.out/'config.json').write_text(json.dumps(config,indent=2));print(config,flush=True)
    if a.arm=='head':print('trainable roots: head_hidden, head_logits',flush=True)
    initial_params=params; best_params=params; rng=np.random.default_rng(a.seed); key=jax.random.PRNGKey(a.seed)
    plateau=Plateau(Patience(lr_drop=700,early_stop=1750,unit='steps',smoothing_steps=350),int(np.ceil(len(xf)/2)))
    rows=[]; start=time.monotonic(); learning_rate=a.lr
    for epoch in range(1,a.max_epochs+1):
        t=time.monotonic(); order=rng.permutation(len(xf)); losses=[]
        for i in range(0,len(order),2):
            ix=order[i:i+2]
            if len(ix)<2:ix=np.r_[ix,order[0]]
            key,sub=jax.random.split(key);params,state,val=step(params,state,jnp.asarray(xf[ix]),jnp.asarray(yf[ix]),sub,jnp.asarray(learning_rate));losses.append(float(val))
        pe=predict(params,xe); score=kappa(pe,ye); decision=plateau.update(score)
        if score>best:best,best_epoch,best_params=score,epoch,params;save(params,epoch)
        if decision.drop_lr:learning_rate*=0.5
        row=dict(lr=learning_rate,epoch=epoch,step=plateau.step,loss=float(np.mean(losses)),eval_kappa=score,best_kappa=best,best_epoch=best_epoch,seconds=time.monotonic()-t)
        rows.append(row)
        with (a.out/'epochs.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(rows)
        print(row,flush=True)
        if decision.stop:break
    if a.arm=='head':
        frozen=[k for k in params if k not in {'head_hidden','head_logits'}]
        for k in frozen:
            assert all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(jax.tree.leaves(params[k]),jax.tree.leaves(initial_params[k])))
    pred_dir=a.out/'predictions';pred_dir.mkdir(exist_ok=True)
    for role in ('eval','test','guard'):
        for name in plan[role]:
            with np.load(a.data/'nights'/f'{name}.npz') as r:
                base=r['x'][None];x=view(r)[None];n=int(r['n']);truth=r['truth'][:n] if 'truth' in r else r['y'][:n]
                p=predict(best_params,x)[0,:n];p0=np.asarray(jax.nn.softmax(release.infer(base),axis=-1))[0,:n]
                np.savez_compressed(pred_dir/f'{name}.npz',probabilities=p,reference=p0,y=truth,missing=r['common_missing'][:n],still=r['still'][:n],anglez=r['anglez'][:n],role=role,transition=(np.r_[False,truth[1:]!=truth[:-1]]|np.r_[truth[1:]!=truth[:-1],False]))
    (a.out/'complete.json').write_text(json.dumps(dict(best_epoch=best_epoch,best_eval_kappa=best,seconds=time.monotonic()-start,smoke=a.smoke),indent=2))
    print('complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('out',type=Path);p.add_argument('--fold',choices=list('ABCDE'),default='E');p.add_argument('--arm',choices=['head','full','land'],default='full');p.add_argument('--seed',type=int,default=0);p.add_argument('--lr',type=float,default=0.001);p.add_argument('--max-epochs',type=int,default=100);p.add_argument('--smoke',action='store_true');p.add_argument('--layout',choices=list(LAYOUTS),required=True);p.add_argument('--pooling',choices=['mean','cnn'],required=True);main(p.parse_args())
