"""Do not compare a five-draw mean against a conveniently selected single draw."""
import json
import pandas as pd
from comparison import collect
from revision_matrix import jobs


def add(root,arm,draw,value,capped=False):
    out=root/arm/f'draw{draw}_seed0';(out/'analysis').mkdir(parents=True)
    (out/'config.json').write_text(json.dumps(dict(arm=arm,draw=draw,seed=0,parameters=2_000_000)))
    (out/'complete.json').write_text(json.dumps(dict(time_capped=capped,train_seconds=10)))
    (out/'analysis/report_complete.json').write_text('{}')
    pd.DataFrame([dict(cohort=c,kappa_4stage=value) for c in ['SleepAccel','SleepAccel-Clinical']]).to_csv(out/'analysis/summary.csv',index=False)


def test_compare_only_matched_completed_draws(tmp_path):
    add(tmp_path,'control1',0,.4);add(tmp_path,'control1',1,.1);add(tmp_path,'final7',0,.35)
    df,draws,coverage=collect(tmp_path)
    assert draws==[0] and coverage['control1']==[0,1]
    assert df[df.experiment=='control1'].kappa_4stage.mean()==.4
    add(tmp_path,'final7',1,.2)
    assert collect(tmp_path)[1]==[0,1]


def test_capped_fit_does_not_extend_coverage(tmp_path):
    add(tmp_path,'control1',0,.4);add(tmp_path,'control1',1,.1,capped=True);add(tmp_path,'final7',0,.35)
    assert collect(tmp_path)[2]['control1']==[0]


def test_matrix_preserves_every_architecture_and_draw():
    plan=jobs();assert len(plan)==50
    for arm in {j['arm'] for j in plan}:
        assert {j['draw'] for j in plan if j['arm']==arm}==set(range(5))
    assert {'final7','exchange7','shared7','control1_5m','final5_5m','exchange5_5m','shared5_5m'} <= {j['arm'] for j in plan}
