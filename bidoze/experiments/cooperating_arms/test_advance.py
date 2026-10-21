"""Protect the compute continuation against averaging away a cohort regression."""
import json
import pandas as pd
from advance_counts import choose


def populate(root,scores,capped=None):
    for arm,(sleep,clinical) in scores.items():
        out=root/arm/'draw0_seed0';(out/'analysis').mkdir(parents=True)
        (out/'complete.json').write_text(json.dumps(dict(time_capped=arm==capped)))
        pd.DataFrame([dict(cohort='SleepAccel',kappa_4stage=sleep),
                      dict(cohort='SleepAccel-Clinical',kappa_4stage=clinical)]).to_csv(out/'analysis/summary.csv',index=False)


def test_prefer_shared_representation_when_differences_are_small(tmp_path):
    populate(tmp_path,dict(control1=(.4,.3),final7=(.42,.30),exchange7=(.415,.305),shared7=(.41,.31)))
    result=choose(tmp_path)
    assert result['selected']=='shared7'
    assert result['arms']==['shared3','shared5']


def test_do_not_average_away_large_clinical_regression(tmp_path):
    populate(tmp_path,dict(control1=(.4,.3),final7=(.46,.24),exchange7=(.35,.25),shared7=(.35,.25)))
    assert choose(tmp_path)['action']=='review'


def test_capped_control_does_not_trigger_more_compute(tmp_path):
    populate(tmp_path,dict(control1=(.4,.3),final7=(.42,.31),exchange7=(.42,.31),shared7=(.42,.31)),capped='control1')
    assert choose(tmp_path)['action']=='review'
