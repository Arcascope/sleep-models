"""Training-only channel ordering, joint PCA and exact pooling-CNN inspection."""
from __future__ import annotations
import argparse,csv,hashlib,html,json,os,subprocess,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import linen as nn
from safetensors.numpy import load_file
from scipy.cluster.hierarchy import linkage,optimal_leaf_ordering,leaves_list
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from arcasleep_bidoze.model.definition import FramePoolModule
from pisces_lite.proc.constants import SPECTROGRAM_PADDING_VALUE as PAD
from boat_windows.prepare import LAYOUTS

ARMS=['native_cnn','contained_cnn','nonoverlap_cnn'];DB=10/np.log(10.)

def load_pool(run,arm):
    bundle=run/f'training/{arm}/D/seed0/bundle'
    weights=load_file(bundle/'model.safetensors');kernel=weights['frame_pool/conv/kernel'];bias=weights['frame_pool/conv/bias']
    assert kernel.shape==(15,121,64)
    module=FramePoolModule(15,64,expected_frames=len(LAYOUTS[arm.rsplit('_',1)[0]]))
    params={'conv':{'kernel':jnp.asarray(kernel),'bias':jnp.asarray(bias)}}
    @jax.jit
    def apply(x):
        pooled,state=module.apply({'params':params},x,capture_intermediates=lambda mdl,method:isinstance(mdl,nn.Conv),mutable=['intermediates'])
        return pooled[:,:,-64:],state['intermediates']['conv']['__call__'][0]
    return apply,kernel,bias

def extract(a,meta):
    pools={arm:load_pool(a.run,arm) for arm in ARMS};audits=[];(a.out/'features').mkdir(exist_ok=True)
    for i,name in enumerate(meta,1):
        with np.load(a.run/'setup/data/nights'/f'{name}.npz') as z:
            n=int(z['n']);frames=z['frames'].reshape(n,15,60,2);common=z['common_missing'].copy()
        result={'common_missing':common};errors={}
        for arm,(apply,kernel,bias) in pools.items():
            layout=arm.rsplit('_',1)[0];masked=frames.copy();masked[:,[j for j in range(15) if j not in LAYOUTS[layout]]]=PAD
            flat=masked.reshape(n,15,120);real=~np.all(flat==PAD,axis=2);x=np.where(real[:,:,None],flat,0.)
            count=real.sum(1);mean=x.sum(1)/np.maximum(count[:,None],1)
            # Exact linear decomposition, conditional on the actual valid mask.
            weight_sum=np.einsum('ep,pfk->efk',real.astype(np.float32),kernel[:,:120],optimize=True)
            average=np.einsum('ef,efk->ek',mean,weight_sum,optimize=True)
            contrast=np.einsum('epf,pfk->ek',np.where(real[:,:,None],x-mean[:,None,:],0.),kernel[:,:120],optimize=True)
            presence=real.astype(np.float32)@kernel[:,120,:]+bias
            activation=[];pre=[]
            for start in range(0,n,256):
                block=np.full((256,15,60,2),PAD,dtype=np.float32);take=min(256,n-start);block[:take]=masked[start:start+take]
                h,u=apply(jnp.asarray(block.reshape(1,3840,60,2)));activation.append(np.asarray(h)[0,:take]);pre.append(np.asarray(u)[0,:take])
            activation=np.concatenate(activation);pre=np.concatenate(pre)
            err=float(np.max(np.abs(pre-(average+contrast+presence))))
            assert np.allclose(pre,average+contrast+presence,rtol=2e-4,atol=2e-4),(name,arm,err)
            result[arm]=activation;result[arm+'_pre']=pre;result[arm+'_mean']=average.astype(np.float32);result[arm+'_contrast']=contrast.astype(np.float32);result[arm+'_presence']=presence.astype(np.float32)
            errors[arm]=err
        np.savez_compressed(a.out/'features'/f'{name}.npz',**result)
        audits.append(dict(name=name,epochs=n,max_decomposition_error=max(errors.values())))
        if i%20==0 or i==len(meta):print('extracted',i,'/',len(meta),flush=True)
    pd.DataFrame(audits).to_csv(a.out/'extraction_audit.csv',index=False)
    return pools

def fit_basis(a,fit,meta):
    # Equal participant mass; equal night mass within participant; equal valid epoch mass per night.
    persons={name:(meta[name]['cohort'],meta[name]['participant']) for name in fit};counts={p:list(persons.values()).count(p) for p in set(persons.values())}
    mean=np.zeros(192);second=np.zeros((192,192));mass=0.;component_stats=[]
    for name in fit:
        with np.load(a.out/'features'/f'{name}.npz') as z:
            take=~z['common_missing'];values=np.concatenate([z[arm] for arm in ARMS],axis=1)[take].astype(np.float64)
            assert len(values)>0
            weight=1/(len(counts)*counts[persons[name]])
            mean+=weight*values.mean(0);second+=weight*(values.T@values)/len(values);mass+=weight
            for arm in ARMS:
                pre=z[arm+'_pre'][take];terms=[z[arm+'_'+term][take] for term in ('mean','contrast','presence')]
                component_stats.append(dict(name=name,arm=arm,weight=weight,pre_sd=float(np.sqrt(np.mean(np.var(pre,axis=0)))),mean_sd=float(np.sqrt(np.mean(np.var(terms[0],axis=0)))),contrast_sd=float(np.sqrt(np.mean(np.var(terms[1],axis=0)))),presence_sd=float(np.sqrt(np.mean(np.var(terms[2],axis=0)))),contrast_rms=float(np.sqrt(np.mean(terms[1]**2))),mean_rms=float(np.sqrt(np.mean(terms[0]**2))),presence_rms=float(np.sqrt(np.mean(terms[2]**2)))))
    assert abs(mass-1)<1e-10
    cov=second-np.outer(mean,mean);sd=np.sqrt(np.maximum(np.diag(cov),0));active=sd>max(1e-8,float(np.median(sd[sd>0]))*1e-6)
    scale=np.where(active,sd,1.)
    display_scale=scale.copy()
    for i in range(3):
        sl=slice(i*64,(i+1)*64);fallback=float(np.median(sd[sl][active[sl]]));display_scale[sl]=np.where(active[sl],sd[sl],fallback)
    corr=cov/np.outer(scale,scale);corr[~active,:]=0;corr[:,~active]=0;corr=(corr+corr.T)/2
    vals,vectors=np.linalg.eigh(corr);order=np.argsort(vals)[::-1];vals=np.maximum(vals[order],0);vectors=vectors[:,order]
    for j in range(192):
        if vectors[np.argmax(np.abs(vectors[:,j])),j]<0:vectors[:,j]*=-1
    orders=[]
    for i,arm in enumerate(ARMS):
        block=np.clip(corr[i*64:(i+1)*64,i*64:(i+1)*64],-1,1);distance=1-block;np.fill_diagonal(distance,0)
        condensed=squareform(distance,checks=False);tree=linkage(condensed,method='average');orders.append(leaves_list(optimal_leaf_ordering(tree,condensed)))
    np.savez_compressed(a.out/'basis.npz',mean=mean,scale=scale,display_scale=display_scale,active=active,vectors=vectors,eigenvalues=vals,orders=np.array(orders))
    table=pd.DataFrame(component_stats);table.to_csv(a.out/'fit_linear_components.csv',index=False)
    summary=[]
    for arm,g in table.groupby('arm'):
        summary.append(dict(arm=arm,**{key:float(np.sqrt(np.sum(g.weight*g[key]**2))) for key in ('pre_sd','mean_sd','contrast_sd','presence_sd','contrast_rms','mean_rms','presence_rms')}))
    pd.DataFrame(summary).to_csv(a.out/'fit_component_summary.csv',index=False)
    pd.DataFrame({'PC':np.arange(1,193),'variance_fraction':vals/max(vals.sum(),1e-12),'cumulative_fraction':np.cumsum(vals)/max(vals.sum(),1e-12)}).to_csv(a.out/'pca_variance.csv',index=False)
    pd.DataFrame([dict(arm=arm,row=row,channel=int(ch),fit_mean=mean[i*64+ch],fit_sd=sd[i*64+ch],active=bool(active[i*64+ch])) for i,arm in enumerate(ARMS) for row,ch in enumerate(orders[i])]).to_csv(a.out/'channel_order.csv',index=False)
    return dict(mean=mean,scale=scale,display_scale=display_scale,active=active,vectors=vectors,eigenvalues=vals,orders=np.array(orders))

def atlases(a,pools,basis):
    plt.rcParams.update({'font.size':9})
    for i,arm in enumerate(ARMS):
        _,w,b=pools[arm];positions=LAYOUTS[arm.rsplit('_',1)[0]];weights=w[positions,:120].reshape(len(positions),60,2,64)
        bound=float(np.percentile(np.abs(weights),99));fig,axs=plt.subplots(8,8,figsize=(24,20))
        for row,(ax,ch) in enumerate(zip(axs.flat,basis['orders'][i])):
            image=np.concatenate([weights[:,:,0,ch],weights[:,:,1,ch]],axis=0)
            ax.imshow(image,aspect='auto',origin='lower',cmap='coolwarm',vmin=-bound,vmax=bound,extent=(.05,6.05,0,2*len(positions)))
            ax.axhline(len(positions),color='#333',lw=.4);ax.set_title(f'row {row}: channel {ch}');ax.set_yticks([]);ax.set_xticks([1,3,6]);ax.tick_params(labelsize=7)
        fig.suptitle(f'{arm}: learned spectral weights, ordered by training activation correlation\nColumns: 0.1–6 Hz; lower rows magnitude, upper rows jerk; selected window starts {str([2*p for p in positions])}\nSigned weights at shared model-wide ±99th-percentile scale. Presence weights and bias retained separately; inactive slots omitted.',fontsize=14)
        fig.tight_layout(rect=(0,0,1,.94));fig.savefig(a.out/f'{arm}_filters.png',dpi=130);plt.close(fig)
        np.savez_compressed(a.out/f'{arm}_weights.npz',kernel=w,bias=b,selected_positions=positions)
    fig,ax=plt.subplots(figsize=(10,4));v=basis['eigenvalues'];ax.plot(np.arange(1,193),np.cumsum(v)/v.sum());ax.set_xlim(1,64);ax.set_ylim(0,1);ax.set_xlabel('Joint principal components (first 64 shown)');ax.set_ylabel('Cumulative training variance fraction');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(a.out/'pca_variance.png',dpi=150);plt.close(fig)

def render(job):
    run,out,name,role,cohort,basis=job;run=Path(run);out=Path(out);npcs=12
    with np.load(run/'setup/data/nights'/f'{name}.npz') as z:
        n=int(z['n']);surfaces={layout:z['x_'+layout][:n,:60].copy() for layout in LAYOUTS};missing=z['common_missing'].copy()
        physical_missing={layout:z['x_'+layout][:n,-1,0]>=1 for layout in LAYOUTS}
        frames=z['frames'].reshape(n,15,60,2);valid=~np.all(frames==PAD,axis=(2,3));limits=[np.percentile(frames[:,:,:,c][valid]*DB,[1,99]) for c in range(2)]
    with np.load(out/'features'/f'{name}.npz') as z:features={key:z[key].copy() for key in z.files}
    fig,axs=plt.subplots(6,3,figsize=(25,20),sharex=True,gridspec_kw={'height_ratios':[1.4,1.4,2.5,1.7,1.3,1.3]})
    pcscale=np.sqrt(np.maximum(basis['eigenvalues'][:npcs],1e-12));cmap=matplotlib.colormaps['RdBu_r'].copy();cmap.set_bad('#888888')
    for i,arm in enumerate(ARMS):
        layout=arm.rsplit('_',1)[0];offset=slice(i*64,(i+1)*64);order=basis['orders'][i]
        for c in range(2):
            spec=surfaces[layout][:,:,c].T*DB;spec[:,physical_missing[layout]]=np.nan
            im=axs[c,i].imshow(spec,origin='lower',aspect='auto',extent=(0,n/120,.05,6.05),vmin=limits[c][0],vmax=limits[c][1],interpolation='nearest');axs[c,i].set_ylabel(('Magnitude' if c==0 else 'Jerk')+' (Hz)')
            if i==2:fig.colorbar(im,ax=axs[c,i],fraction=.018,pad=.015,label='cached PSD dB')
        display=(features[arm]-basis['mean'][offset])/basis['display_scale'][offset]
        channels=display[:,order].T
        centered=(features[arm]-basis['mean'][offset])/basis['scale'][offset];centered[:,~basis['active'][offset]]=0
        im=axs[2,i].imshow(channels,origin='lower',aspect='auto',extent=(0,n/120,-.5,63.5),cmap=cmap,vmin=-3,vmax=3,interpolation='nearest');axs[2,i].set_ylabel('64 actual channels\nfit-correlation order; z')
        if i==2:fig.colorbar(im,ax=axs[2,i],fraction=.018,pad=.015,label='training z / fallback scale')
        pc=(centered@basis['vectors'][offset,:npcs]/pcscale).T
        im=axs[3,i].imshow(pc,origin='lower',aspect='auto',extent=(0,n/120,.5,npcs+.5),cmap=cmap,vmin=-1.5,vmax=1.5,interpolation='nearest');axs[3,i].set_yticks([1,4,8,12]);axs[3,i].set_ylabel('Joint PC contributions\ncommon axes; joint-SD units')
        if i==2:fig.colorbar(im,ax=axs[3,i],fraction=.018,pad=.015,label='joint PC SD units')
        # The last two rows use within-night z solely to compare shape.
        # Absolute temporal SD/RMS values are in the fit component CSV.
        for axis,term in [(axs[4,i],'mean'),(axs[5,i],'contrast')]:
            value=features[arm+'_'+term].astype(float)
            value=(value-value.mean(0))/np.maximum(value.std(0),1e-8)
            value=value[:,order].T
            im=axis.imshow(value,origin='lower',aspect='auto',extent=(0,n/120,-.5,63.5),cmap=cmap,vmin=-3,vmax=3,interpolation='nearest');axis.set_ylabel(term+' linear term\nwithin-night channel z')
            if i==2:fig.colorbar(im,ax=axis,fraction=.018,pad=.015,label='within-night z (shape only)')
        axs[0,i].set_title(arm,fontsize=14)
        for ax in axs[:,i]:
            ax.set_xlim(0,n/120)
            edges=np.diff(np.r_[False,missing,False].astype(int))
            for lo,hi in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):ax.axvspan(lo/120,hi/120,color='#777',alpha=.15)
        axs[-1,i].set_xlabel('Elapsed hours from PSG origin')
    variance=basis['eigenvalues'];fraction=variance[:npcs].sum()/variance.sum()
    fig.suptitle(f'{name} | {cohort} / {role} | saved pooling CNNs, no retraining',fontsize=16)
    fig.text(.5,.008,f'Training-only channel order and participant/night-balanced joint PCA. First {npcs} joint PCs explain {fraction:.1%} training standardized-feature variance.\nPC panels show each model block contribution; summing the three gives the joint PC score. Channel z scales ±3; PC contributions ±1.5 joint SD.\nLinear mean/contrast panels are BEFORE GELU and standardized within this night for shape only; their amplitudes are not comparable. Read component CSVs for magnitude.\nWhite ≈ zero standardized response; grey vertical overlays mark unavailable coverage; physically missing spectra are pale.\nTraining-unvarying channels use a typical-active-SD fallback for display; joint PCA omits unsupported training directions.\nChannels are latent features, not sleep stages; signs/order/PCA variance alone do not establish physiological meaning.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.065,1,.965));path=out/f'{name}.png';temp=path.with_suffix('.png.partial');fig.savefig(temp,dpi=130,format='png');plt.close(fig);os.replace(temp,path)
    return dict(name=name,role=role,cohort=cohort,image=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def main(a):
    assert a.out.name!='hypnograms','Use a separate diagnostic directory'
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'config.json').write_text(json.dumps(dict(pid=os.getpid(),arm='cnn_inspection',fold='D',seed=0)))
    meta={r['name']:r for r in csv.DictReader((a.run/'setup/data/inventory.csv').open())};plan=json.loads((a.run/'training/native_cnn/D/seed0/split.json').read_text());fit=plan['fit'];roles={name:role for role in ('test','eval','guard') for name in plan[role]}
    assert not set(fit)&set(roles)
    if a.reuse_features:
        assert all((a.out/'features'/f'{name}.npz').exists() for name in meta)
        pools={arm:load_pool(a.run,arm) for arm in ARMS}
    else:pools=extract(a,meta)
    basis=fit_basis(a,fit,meta);atlases(a,pools,basis)
    # Verify additive shared-axis convention on an arbitrary saved training night.
    with np.load(a.out/'features'/f'{fit[0]}.npz') as z:
        joint=np.concatenate([z[arm] for arm in ARMS],axis=1);joint=(joint-basis['mean'])/basis['scale'];joint[:,~basis['active']]=0
        parts=sum(joint[:,i*64:(i+1)*64]@basis['vectors'][i*64:(i+1)*64,:12] for i in range(3));assert np.allclose(parts,joint@basis['vectors'][:,:12],atol=1e-8)
    jobs=[(str(a.run),str(a.out),name,role,meta[name]['cohort'],basis) for name,role in roles.items()];manifest=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,future in enumerate(as_completed([pool.submit(render,job) for job in jobs]),1):
            manifest.append(future.result());(a.out/'progress.json').write_text(json.dumps(dict(recordings=i,total=len(jobs))))
            if i%10==0 or i==len(jobs):print('rendered',i,'/',len(jobs),flush=True)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    components=pd.read_csv(a.out/'fit_component_summary.csv')
    stats=''.join(f'<tr><td>{row.arm}</td><td>{row.mean_sd:.2f}</td><td>{row.contrast_sd:.2f}</td><td>{row.pre_sd:.2f}</td></tr>' for row in components.itertuples())
    h=html.escape;links=' '.join(f'<a href="{arm}_filters.png">{arm} filter atlas</a>' for arm in ARMS)
    records=''.join(f'<tr><td><a href="{r["image"]}">{h(r["name"])}</a></td><td>{h(r["cohort"])}</td><td>{h(r["role"])}</td></tr>' for r in sorted(manifest,key=lambda r:(r['role'],r['name'])))
    (a.out/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>Pooling CNN inspection</title><style>body{{font:16px system-ui;margin:25px}}td{{padding:7px}}a{{margin-right:15px}}</style><a href="../">Existing hypnograms</a><h1>Pooling CNN inspection</h1><p>Separate diagnostic plots for the 64 learned pooling channels. Existing hypnograms are unchanged.</p><p>{links} <a href="pca_variance.png">PCA variance</a></p><p>Each recording compares native, contained and nonoverlap inputs, the reordered actual channels, shared joint-PCA contributions and exact pre-GELU mean/temporal-contrast terms. Ordering and basis use fit recordings only, equally weighting participants and nights. No PSG labels select the basis or channel order.</p><p>Channel panels use training z scales; training-unvarying channels are retained with a typical-active-SD fallback scale and marked in channel_order.csv. PCA excludes those unsupported directions. White means near-zero standardized response; grey vertical overlays mark unavailable coverage. the last two panels use within-night z scales for shape only. Joint PCs are shared across models, not separate arbitrary PCA axes. Their block contributions sum to the joint score. These latent directions are not physiological labels or a causal attribution to predictions.</p><h2>What the linear filters respond to</h2><p>The kernel spans one output epoch. Each learned channel is a signed template over window position × frequency × magnitude/jerk, plus window-presence weights and bias, followed by GELU. Red/blue weights are signed coefficients, not wake/sleep labels.</p><p>Training RMS of within-night temporal SD across channels, equally weighted by participant/night. Terms can correlate, so their SDs do not add; these values do not attribute classification accuracy. The presence/bias remainder is saved in the CSV.</p><table><tr><th>Model</th><th>Mean spectral term SD</th><th>Within-epoch contrast SD</th><th>Complete preactivation SD</th></tr>{stats}</table><p><a href="channel_order.csv">Channel row → original unit IDs</a><a href="fit_component_summary.csv">Component measurements</a><a href="pca_variance.csv">PCA variance fractions</a></p><h2>All evaluation recordings</h2><table><tr><th>Recording</th><th>Cohort</th><th>Role</th></tr>{records}</table>''')
    protocol=dict(code=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),models=ARMS,fit_records=len(fit),plot_records=len(jobs),fit_weighting='equal participant mass, equal night mass within participant, equal common-valid epoch mass within night',ordering='training activation Pearson correlation distance, average linkage, optimal leaf ordering; original channel IDs in channel_order.csv',basis='PCA of joint concatenated 3x64 training-standardized activations; deterministic loading sign; block contributions sum to joint PCA score; first12 whitened by joint eigenvalue SD',display_inactive='actual-channel plots retain training-unvarying channels using typical-active-SD fallback; joint PCA zeroes unsupported training directions',labels='never used in extraction, ordering or basis',components='exact pre-GELU decomposition into valid-window mean, zero-mean within-epoch contrasts, and presence+bias; conditional on actual valid mask; not post-GELU causal attribution',unchanged='existing hypnograms, training inputs, parameters and predictions',checkpoints={arm:hashlib.sha256((a.run/f'training/{arm}/D/seed0/bundle/model.safetensors').read_bytes()).hexdigest() for arm in ARMS})
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2));(a.out/'complete.json').write_text(json.dumps(dict(recordings=len(jobs),plots=len(manifest)+4)))
    print('CNN INSPECTION COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('out',type=Path);p.add_argument('--workers',type=int,default=8);p.add_argument('--reuse-features',action='store_true');main(p.parse_args())
