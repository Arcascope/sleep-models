import argparse
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);a=p.parse_args()
data=pd.read_csv(a.out/"summary_physical.csv")
fig,axes=plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True)
rng=np.random.default_rng(713)
groups=[("Eric-Galaxy","Galaxy Watch"),("Eric-iPhone","iPhone")]
colors=["#1f77b4","#c45a23"]
for y,((device,label),color) in enumerate(zip(groups,colors)):
 d=data[data.device==device];jitter=rng.uniform(-.11,.11,len(d))
 for ax,column in zip(axes,["nufft_sum_median_bin_db_q10","nufft_physical_vector_band_rms_mg_q10"]):
  x=d[column].to_numpy()
  ax.scatter(x,y+jitter,s=23,color=color,alpha=.75)
  ax.scatter(np.median(x),y,marker="D",s=55,color="black",zorder=5)
axes[0].axvline(-62.9,ls="--",color="#666666",lw=1,label="SURF Galaxy reference")
axes[0].legend(fontsize=8,loc="lower left")
axes[0].set_xlabel("2–6 Hz floor density (dB re g²/Hz)\nSummed axes; senpy two-sided convention")
axes[1].set_xlabel("Physical vector band RMS (mg)\nSelected quiet frames; includes residual motion")
axes[1].set_xscale("log")
axes[1].set_xlim(.45,6)
axes[1].set_xticks([.5,1,2,5],["0.5","1","2","5"])
from matplotlib.ticker import NullLocator
axes[1].xaxis.set_minor_locator(NullLocator())
for ax in axes:
 ax.set_yticks([0,1],["Galaxy: 21 windows","iPhone: 22 windows"])
 ax.set_ylim(1.4,-.4);ax.grid(axis="x",alpha=.2)
fig.suptitle("Matched device-floor measurements: one person, separate periods",fontsize=11)
target=a.out/"figures";target.mkdir(exist_ok=True)
fig.savefig(target/"matched-device-floor-audit.png",dpi=160)
