import copy
import json
import pytest
from finetuning.selection import REQUIRED_FAMILIES, retention_violations, improves, validate_thresholds


def metrics():
    return dict(typed_acc=.7, typed_ce=.8, retention_acc=.9,
                families={f'retention/{k}':dict(accuracy=.9, ordinal_mae=.2) for k in REQUIRED_FAMILIES})


def test_policy_boundaries_and_serialized_resume():
    base=metrics(); current=copy.deepcopy(base)
    current['retention_acc'] -= .005
    current['families']['retention/nli']['accuracy'] -= .01
    assert not retention_violations(base,current,.005,.01,.05)
    current['families']['retention/score']['ordinal_mae'] += .051
    expected=retention_violations(base,current,.005,.01,.05)
    assert expected
    assert retention_violations(json.loads(json.dumps(base)),current,.005,.01,.05)==expected


def test_missing_and_nonfinite_fail_closed():
    for value in [float('nan'), float('inf'), -.1]:
        with pytest.raises(ValueError): validate_thresholds(value)
    base=metrics(); current=metrics(); del current['families']['retention/score']
    with pytest.raises(ValueError): retention_violations(base,current,.005,.01,.05)
    current=metrics();current['typed_acc']=float('nan')
    with pytest.raises(ValueError): retention_violations(base,current,.005,.01,.05)


def test_accuracy_then_ce():
    assert improves(.71,.9,.7,.8)
    assert improves(.7,.79,.7,.8)
    assert not improves(.69,.1,.7,.8)
    assert not improves(.7,.8,.7,.8)
