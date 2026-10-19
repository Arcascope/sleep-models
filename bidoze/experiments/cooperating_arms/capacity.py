"""Count instantiated parameters and diagram where the fixed budget is spent."""
import argparse
import json
from pathlib import Path
import jax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from model import CooperatingModel
from run import configuration


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();out=a.run/'setup';out.mkdir(exist_ok=True)
    counts={}
    for name in ['control1','final7','exchange7','shared7']:
        cfg,spec=configuration(name);m=CooperatingModel(cfg,spec);m._ensure_built((8,61,2))
        categories={k:0 for k in ['CNN','Input projections','Transformer','Exchange','Fusion and heads']}
        for root,tree in m.params.items():
            n=sum(int(p.size) for p in jax.tree.leaves(tree))
            if 'cnn' in root:key='CNN'
            elif root.endswith('_input'):key='Input projections'
            elif root.startswith('exchange_'):key='Exchange'
            elif '_fwd_' in root or '_rev_' in root or root.endswith('_out') or root.endswith('_out_norm'):key='Transformer'
            else:key='Fusion and heads'
            categories[key]+=n
        counts[name]=categories
    (out/'parameter_counts.json').write_text(json.dumps(counts,indent=2))
    fig,ax=plt.subplots(figsize=(11,5));left=np.zeros(len(counts))
    for key,color in zip(next(iter(counts.values())),['#5b9bd5','#93c7ee','#70ad47','#d47771','#b5a1ce']):
        vals=np.array([x[key] for x in counts.values()])/1e6
        ax.barh(list(counts),vals,left=left,label=key,color=color);left+=vals
    for i,total in enumerate(left):ax.text(total+.02,i,f'{total:.3f}M',va='center')
    ax.set(xlabel='Trainable parameters (millions)',xlim=(0,2.65),title='The same overall budget buys different transformer capacity')
    ax.legend(loc='lower center',bbox_to_anchor=(.5,-.28),ncol=3)
    fig.tight_layout(rect=[0,.08,1,1]);fig.savefig(out/'capacity.png',dpi=160);plt.close(fig)
    print(json.dumps(counts,indent=2))


if __name__=='__main__':main()
