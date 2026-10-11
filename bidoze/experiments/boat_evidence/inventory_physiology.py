"""Inventory independent physiological measurements; vendor sleep-stage exports are not physiology."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pandas as pd


def main(a):
    rows=[]; seen=set()
    for cohort in ('SURF','Somerset'):
        base=a.root/cohort
        for p in sorted(base.rglob('*')):
            if not p.is_file() or p.suffix.lower() not in ('.csv','.json','.xml'):continue
            low=p.name.lower()
            if not any(k in low for k in ('heartrate','heart_rate','heart.rate','ppg','hrv','heartbeat')):continue
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            if digest in seen: continue
            seen.add(digest)
            relative=str(p.relative_to(base)); row=dict(cohort=cohort,file=relative,bytes=p.stat().st_size,kind='measurement candidate',rows='',columns='')
            if p.suffix.lower()=='.csv':
                data=pd.read_csv(p,sep=None,engine="python");row.update(rows=len(data),columns='|'.join(map(str,data.columns)))
            elif p.suffix.lower()=='.json':
                data=json.loads(p.read_text());row.update(rows=len(data),columns='|'.join(data[0]) if isinstance(data,list) and data and isinstance(data[0],dict) else '|'.join(data) if isinstance(data,dict) else '')
            rows.append(row)
    a.out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows,columns=['cohort','file','bytes','kind','rows','columns']).to_csv(a.out/'physiology_inventory.csv',index=False)
    (a.out/'physiology_inventory.json').write_text(json.dumps(dict(candidate_files=len(rows),cohorts={c:sum(r['cohort']==c for r in rows) for c in ('SURF','Somerset')},status='availability only; per-night coverage and clock alignment not yet validated',excluded='cleaned_oura TIMESTAMP/OURA files are vendor sleep labels, not HR'),indent=2))
    print(json.loads((a.out/'physiology_inventory.json').read_text()),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('out',type=Path);main(p.parse_args())
