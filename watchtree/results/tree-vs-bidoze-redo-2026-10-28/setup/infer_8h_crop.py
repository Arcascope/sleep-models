"""8 h BiDoze window logits for the PSG-cropped input of the 92 Wearanize+ test nights (the 8 h input of section 1's 'PSG crop' arm).
Same checkpoints, window (960), stride (120), padding and taper as blend-exp2a; only the crop differs. Saves the window logits locally."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import sys, time
import numpy as np
sys.path.insert(0, '/home/eric/Engineering/Work/autopisces/sleep-models-uncropped-loaders/bidoze/results/blend-exp2a-2026-10-28/setup')
import blend_exp2a as B
import jax, jax.numpy as jnp
from arcasleep.data.batching import pad_pooled
from arcasleep.predict import window_taper
RR = '/home/eric/Engineering/Work/autopisces/sleep-models-instinct-trees/watchtree/results/tree-vs-bidoze-redo-2026-10-28'
dest_dir = f'{RR}/local/windows_8h_crop'; os.makedirs(dest_dir, exist_ok=True)
taper = window_taper(B.L)
for k in range(5):
    recs = sorted(B.PRIOR.glob(f'probabilities/test_f{k}_wearanize_*_8h_psg.npz'))
    models = B.load_models(k); module = models[0].module
    f = jax.jit(lambda params, xb: module.apply({'params': params}, xb, jnp.ones(xb.shape[:2], jnp.float32), train=False))
    for p in recs:
        with np.load(p) as z: sid = str(z['subject']); off = int(z['offset']); ny = len(z['y'])
        dest = f'{dest_dir}/test_f{k}_wearanize_{sid}.npz'
        if os.path.exists(dest): continue
        x, y, raw = B.cache(B.UNC / f'wearanize_subject_{sid}_features.npz')
        span = np.flatnonzero(raw != -3); lo, hi = int(span[0]), int(span[-1] + 1)
        assert lo == off and hi - lo == ny, (sid, lo, off, hi - lo, ny)
        xc = x[lo:hi]; n = len(xc)
        starts = np.array(B.window_starts(n, B.L, B.STEP))
        logits = np.zeros((3, len(starts), B.L, 5), np.float32)
        for m, model in enumerate(models):
            for i in range(0, len(starts), 2):
                ss = starts[i:i + 2]
                xb = np.stack([pad_pooled(xc[s:s + B.L], B.L) for s in ss])
                lg = np.asarray(f(model.params, jnp.asarray(xb))); assert np.isfinite(lg).all()
                logits[m, i:i + len(ss)] = lg
        vlen = np.minimum(n - starts, B.L)
        idx = np.arange(len(starts))
        np.savez_compressed(dest, logits=logits, starts=starts, valid=np.arange(B.L)[None] < vlen[:, None], taper=taper, idx_a=idx, idx_b=idx,
                            gap=xc[:, -1, 0] >= 1, y=y[lo:hi], raw_y=raw[lo:hi], subject=sid, cohort='wearanize', role='test', fold=k, n=n, offset=lo)
        B.log(f'crop windows f{k} {sid} n={n}')
print('CROP_DONE', flush=True)
