"""Training-style probability, stage and exact-input figures for every evaluated recording."""
from __future__ import annotations
import argparse,csv,hashlib,html,json,os,subprocess,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FuncFormatter
import numpy as np
from arcasleep.reporting.hypnogram import _draw_stage_panel,_CLASS_NAMES,_CLASS_COLORS,_ORDERED_STAGE_INDICES
from arcasleep_bidoze.adult import hybrid_calls
from arcasleep.ggir import evaluate_epoch_grid
from pisces_lite.proc.constants import SPECTROGRAM_PADDING_VALUE as PAD
from boat_windows.prepare import LAYOUTS
from boat_axes.score_training import DIVISOR

ARMS=['native_mean','contained_mean','nonoverlap_mean','native_cnn','contained_cnn','nonoverlap_cnn']
DB=10/np.log(10.)

def stage(ax,y,label):
    _draw_stage_panel(ax,y,label,class_count=5)
    if ax.get_legend():ax.get_legend().remove()

def render_recording(job):
    run,out,name,role,cohort,dpi=job;run=Path(run);out=Path(out)
    dest=out/name;dest.mkdir(parents=True,exist_ok=True)
    with np.load(run/'setup/data/nights'/f'{name}.npz') as z:
        n=int(z['n']);frames=z['frames'].reshape(n,15,60,2)
        surfaces={layout:z[f'x_{layout}'][:n,:60].copy() for layout in LAYOUTS}
        physical_missing={layout:z[f'x_{layout}'][:n,-1,0]>=1 for layout in LAYOUTS}
        common=z['common_missing'].copy()
    real=~np.all(frames==PAD,axis=(2,3))
    scales=[]
    for c in range(2):
        values=frames[:,:,:,c][real]*DB
        lo,hi=np.percentile(values,[1,99]) if values.size else (-100.,0.)
        if hi<=lo:hi=lo+1
        scales.append((float(lo),float(hi)))
    # Supplemental sheets preserve every selected frame column; a full-night
    # overview necessarily rasterizes many retained windows into fewer pixels.
    for layout in LAYOUTS:
        channels=[]
        positions=LAYOUTS[layout];allowed=real[:,positions]
        for c in (1,0):  # origin='lower': magnitude top, jerk bottom
            lo,hi=scales[c];values=frames[:,:,:,c][:,positions,:].copy();values[~allowed]=np.nan
            values=values.reshape(n*len(positions),60).T*DB
            rgba=matplotlib.colormaps['viridis'](np.clip((values-lo)/(hi-lo),0,1))
            rgba[~np.isfinite(values)]=[.93,.93,.93,1.]
            channels.append(rgba)
        sheet=np.repeat(np.concatenate(channels,axis=0),3,axis=0)
        sheet_path=dest/f'{layout}_frame_inputs.png'
        sheet_tmp=sheet_path.with_suffix('.png.partial')
        plt.imsave(sheet_tmp,sheet,origin='lower',format='png');os.replace(sheet_tmp,sheet_path)
    results=[];hours=np.arange(n+1)/120
    cmap=matplotlib.colormaps['viridis'].copy();cmap.set_bad('#eeeeee')
    for arm in ARMS:
        path=dest/f'{arm}.png'
        with np.load(run/f'training/{arm}/D/seed0/predictions/{name}.npz') as z:
            probabilities=z['probabilities'].copy();truth=z['y'].copy();missing=z['missing'].copy();angle=z['anglez'].copy()
        assert len(truth)==n and probabilities.shape==(n,5)
        assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1),1,atol=1e-5)
        assert np.array_equal(missing,common)
        layout=arm.rsplit('_',1)[0];pooling=arm.rsplit('_',1)[1]
        raw=probabilities.argmax(1);confidence=probabilities.max(1)
        scored=hybrid_calls(probabilities[:,:4],probabilities[:,:4],DIVISOR);scored[missing]=4
        masks=evaluate_epoch_grid(angle);scored[(scored>0)&(scored<4)&~masks.is_sib&~masks.is_spt]=0
        fig,axs=plt.subplots(9,1,figsize=(32,23),sharex=True,gridspec_kw={'height_ratios':[1.8,.8,1.2,1.2,1.2,1.7,1.7,2,2]})
        cumulative=np.zeros(n)
        for k in _ORDERED_STAGE_INDICES:
            upper=cumulative+probabilities[:,k]
            axs[0].fill_between(hours,np.r_[cumulative,cumulative[-1]],np.r_[upper,upper[-1]],step='post',color=_CLASS_COLORS[_CLASS_NAMES[k]],alpha=.85,label=_CLASS_NAMES[k]);cumulative=upper
        axs[0].set_ylim(0,1);axs[0].set_ylabel('Raw softmax\nprobabilities');axs[0].legend(ncol=5,loc='upper right',frameon=False)
        axs[1].stairs(confidence,hours,color='#333');axs[1].set_ylim(0,1);axs[1].set_ylabel('Raw argmax\nconfidence')
        # The training stage helper uses epoch coordinates; transform its artists
        # through an x-axis scale so every panel shares the same hour grid.
        for ax,y,label in [(axs[2],raw,'Raw argmax'),(axs[3],scored,'Release + GGIR'),(axs[4],truth,'PSG truth')]:
            stage(ax,y,label)
            for coll in ax.collections:
                from matplotlib.transforms import Affine2D
                coll.set_transform(Affine2D().scale(1/120,1)+ax.transData)
        for c,label in enumerate(('Magnitude','Difference-norm jerk')):
            lo,hi=scales[c]
            pooled=surfaces[layout][:,:,c].T*DB;pooled[:,physical_missing[layout]]=np.nan
            image=axs[5+c].imshow(pooled,origin='lower',aspect='auto',extent=(0,n/120,.05,6.05),vmin=lo,vmax=hi,cmap=cmap,interpolation='nearest')
            axs[5+c].set_ylabel(label+'\nepoch mean (Hz)');fig.colorbar(image,ax=axs[5+c],pad=.003,fraction=.012,label='cached PSD dB')
            positions=LAYOUTS[layout]
            selected=frames[:,:,:,c][:,positions,:].copy();valid=real[:,positions]
            selected[~valid]=np.nan
            frameplot=selected.reshape(n*len(positions),60).T*DB
            image=axs[7+c].imshow(frameplot,origin='lower',aspect='auto',extent=(0,n/120,.05,6.05),vmin=lo,vmax=hi,cmap=cmap,interpolation='nearest')
            axs[7+c].set_ylabel(label+'\nselected windows (Hz)');fig.colorbar(image,ax=axs[7+c],pad=.003,fraction=.012,label='cached PSD dB')
        # Shade unavailable common coverage; truth itself remains the saved PSG.
        absent=common|(truth<0);edges=np.diff(np.r_[False,absent,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
        for ax in axs:
            for lo,hi in zip(starts,ends):ax.axvspan(lo/120,hi/120,color='#777777',alpha=.18,zorder=10)
            ax.set_xlim(0,n/120);ax.xaxis.set_major_locator(MultipleLocator(1));ax.xaxis.set_minor_locator(MultipleLocator(.25));ax.grid(axis='x',alpha=.16)
        axs[-1].set_xlabel('Elapsed hours from original PSG epoch origin')
        title=f'{name} | {cohort} / {role} | {arm} | selected windows: {len(LAYOUTS[layout])} | selected checkpoint, seed 0'
        notes=('Raw argmax/confidence uses the five-class softmax; scored stages use frozen divisors + physical-gap mask + GGIR.\n'
               'Spectral limits are shared across all six arms for this recording/channel (native frame 1st–99th percentiles).\n'
               'Selected windows are packed contiguously within their assigned 30-s epoch; column width is display spacing, not raw window support. Pale columns mean physically missing windows.\n'
               'Mean models see the epoch surface; CNN models see selected frames and also retain their masked epoch mean. Grey spans: unavailable coverage/unscored truth.')
        fig.suptitle(title,fontsize=15,y=.997);fig.text(.5,.005,notes,ha='center',va='bottom',fontsize=11)
        fig.tight_layout(rect=(0,.052,1,.979));temporary=path.with_suffix('.png.partial');fig.savefig(temporary,dpi=dpi,facecolor='white',format='png');plt.close(fig);os.replace(temporary,path)
        assert path.stat().st_size>10000
        results.append(dict(name=name,arm=arm,role=role,cohort=cohort,epochs=n,image=f'{name}/{arm}.png',bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),spectral_limits_db=scales))
    h=html.escape;links=' '.join(f'<a href="{arm}.png">{h(arm)}</a>' for arm in ARMS)
    frame_links=' '.join(f'<a href="{layout}_frame_inputs.png">{h(layout)} frame inputs</a>' for layout in LAYOUTS)
    options=''.join(f'<option>{h(x)}</option>' for x in ARMS)
    page=f'''<!doctype html><meta charset="utf-8"><title>{h(name)}</title><style>body{{font:16px system-ui;margin:20px}}img{{width:100%}}a{{margin-right:15px}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}select{{font:inherit}} </style><a href="../">All recordings</a><h2>{h(name)}</h2><p>{h(cohort)} / {h(role)}. Click an image for full resolution. Colour limits match across configurations.</p><p>{links}</p><p>Original-resolution input sheets: {frame_links}. One column per selected window, packed within each 30-second epoch; magnitude above jerk, both 0.1–6 Hz. Starts: native 0,2,…,28 s; contained 0,2,…,20 s; nonoverlap 0,10,20 s. All windows are 10 s long. Pale columns are physically missing windows; colour bars are in the full figures.</p><div class="pair"><section><select id="left">{options}</select><a id="la" href="native_mean.png"><img id="li" src="native_mean.png"></a></section><section><select id="right">{options}</select><a id="ra" href="contained_mean.png"><img id="ri" src="contained_mean.png"></a></section></div><script>right.value='contained_mean';for(const s of ['left','right']){{document.getElementById(s).onchange=function(){{const t=s==='left'?'l':'r';document.getElementById(t+'i').src=this.value+'.png';document.getElementById(t+'a').href=this.value+'.png'}}}}</script>'''
    (dest/'index.html').write_text(page)
    return results

def main(a):
    a.out.mkdir(parents=True,exist_ok=True)
    config=dict(pid=os.getpid(),arm='training_hypnograms',fold='D',seed=0,workers=a.workers,dpi=a.dpi)
    (a.out/'config.json').write_text(json.dumps(config,indent=2))
    plan=json.loads((a.run/'training/native_mean/D/seed0/split.json').read_text());meta={r['name']:r for r in csv.DictReader((a.run/'setup/data/inventory.csv').open())}
    roles={name:role for role in ('test','eval','guard') for name in plan[role]}
    names=list(roles)
    if a.limit:names=names[:a.limit]
    for arm in ARMS:
        folder=a.run/f'training/{arm}/D/seed0';assert (folder/'complete.json').exists()
        assert {p.stem for p in (folder/'predictions').glob('*.npz')}==set(roles),'Evaluation IDs differ between arms'
    jobs=[(str(a.run),str(a.out),name,roles[name],meta[name]['cohort'],a.dpi) for name in names]
    manifest=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        pending={pool.submit(render_recording,j):j[2] for j in jobs}
        for i,future in enumerate(as_completed(pending),1):
            manifest.extend(future.result());(a.out/'progress.json').write_text(json.dumps(dict(recordings=i,total=len(jobs),images=len(manifest),seconds=time.monotonic()-start)))
            print(f'rendered {i}/{len(jobs)} recordings, {len(manifest)} figures',flush=True)
    assert len(manifest)==len(names)*6
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    h=html.escape;table=''.join(f'<tr data-search="{h((name+" "+roles[name]+" "+meta[name]["cohort"]).lower())}"><td><a href="{name}/">{h(name)}</a></td><td>{h(meta[name]["cohort"])}</td><td>{h(roles[name])}</td></tr>' for name in names)
    index=f'''<!doctype html><meta charset="utf-8"><title>Window/pooling training hypnograms</title><style>body{{font:16px system-ui;margin:25px;max-width:1400px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #ddd}}input{{font:inherit;padding:8px;width:80%}}</style><h1>Window/pooling training hypnograms</h1><p>{len(names)} recordings × six models = {len(manifest)} full training-style figures. Includes inner eval, held-out vessel test, and Clinical guards.</p><p>Each figure: raw probabilities, argmax confidence, raw predicted stages, scored stages, PSG truth, and magnitude/jerk spectrograms at epoch and selected-frame resolution. Spectral colour limits match across models within each recording/channel. No retraining or threshold selection.</p><p>Eval recordings selected checkpoints and are not independent tests. Start with role <b>test</b> for held-out boats, or <b>guard</b> for Clinical. Selected windows are packed within their assigned epoch; structurally excluded slots are omitted. Pale columns mean physically missing windows. Display column spacing is not raw window duration.</p><input id="search" placeholder="Filter by recording, cohort or role"><table><thead><tr><th>Recording</th><th>Cohort</th><th>Role</th></tr></thead><tbody>{table}</tbody></table><script>search.oninput=()=>{{for(const row of document.querySelectorAll('tr[data-search]'))row.hidden=!row.dataset.search.includes(search.value.toLowerCase())}}</script>'''
    (a.out/'index.html').write_text(index)
    protocol=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),roles={k:sum(roles[n]==k for n in names) for k in ('eval','test','guard')},models=ARMS,probabilities='raw five-class softmax; raw argmax confidence/stages',scored_stages='conditional frozen-divisor stages, common physical missing mask, original GGIR',spectrograms='epoch means and selected original-frame inputs; natural-log PSD converted by 10/ln(10), shared per-recording/channel native p1/p99 limits; gap row excluded',cnn='selected frames are inputs, epoch mean is retained branch; learned 64-dimensional features are not spectra',frame_sheets='three layouts per recording, one pixel per selected window, packed within its epoch; magnitude top/jerk bottom; physically unavailable windows pale; structural slots omitted to prevent downsampling alias; common-mask overlay does not erase real inputs',display_correction='The prior 15-slot raster with structural NaNs aliased into wide pale bands. Only display changed; model inputs, probabilities and metrics unchanged.',time='unchanged PSG origin; no clock correction',scope='all available evaluated predictions; inner eval is selection data',limit=a.limit)
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2));(a.out/'complete.json').write_text(json.dumps(dict(recordings=len(names),figures=len(manifest),seconds=time.monotonic()-start,limited=bool(a.limit)),indent=2))
    print('HYPNOGRAMS COMPLETE',len(manifest),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('out',type=Path);p.add_argument('--workers',type=int,default=8);p.add_argument('--dpi',type=int,default=130);p.add_argument('--limit',type=int,default=0);main(p.parse_args())
