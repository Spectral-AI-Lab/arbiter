"""Recipe budgets, probability preservation and selection boundaries, no network/GPU."""
import json
from types import SimpleNamespace
from unittest.mock import patch
from finetuning.build_80k import TRAIN_QUOTAS, VAL_QUOTAS, Selector, procedural_rows


def row(group,context='example',instruction='Choose.'):
    return dict(id=group,group_id=group,context=context,instruction=instruction,
                question_type='choice',criteria={'a':'A','b':'B'},target={'a':1.,'b':0.})


def test_budget_and_nli_alignment():
    assert sum(TRAIN_QUOTAS.values()) == 80000
    assert sum(VAL_QUOTAS.values()) == 8000
    for quotas in (TRAIN_QUOTAS,VAL_QUOTAS):
        assert all(v%15==0 for k,v in quotas.items() if k.startswith('nli/'))


def test_procedural_targets_are_native():
    source={'id':'test:train:1','state':'A rule.',
            'questions':json.dumps({'q':{'type':'noul','instructions':'Does it apply?'}}),
            'answers':json.dumps({'q':{'noul':.25}})}
    r,=procedural_rows(source,'policy')
    assert r['target']=={'false':.75,'true':.25}
    assert r['synthetic'] is True
    assert r['source_split']=='train'


def test_selector_excludes_test_and_cross_split_contexts(tmp_path):
    held=tmp_path/'test.jsonl';held.write_text(json.dumps({'state':'Reserved text'})+'\n')
    sel=Selector({'train':{'k':2},'validation':{'k':2}},[held],SimpleNamespace(),1024,13)
    with patch('finetuning.build_80k.split_group',side_effect=lambda group,seed: 'validation' if group.startswith('v') else 'train'), \
         patch('finetuning.build_80k.ArbiterModel._prepare_decision',return_value={}):
        sel.add([row('a','reserved TEXT')],['k'])
        assert not sel.rows['train']
        sel.add([row('a','shared')],['k'])
        sel.add([row('v','SHARED')],['k'])
        assert not sel.rows['validation']
        sel.add([row('b','shared')],['k'])
        assert len(sel.rows['train'])==1  # no duplicate-request oversampling
        sel.add([row('v','other')],['k'])
        assert len(sel.rows['validation'])==1


def test_whole_translation_group_not_partially_selected(tmp_path):
    held=tmp_path/'test.jsonl';held.write_text(json.dumps({'state':'reserved'})+'\n')
    sel=Selector({'train':{'k':1},'validation':{'k':1}},[held],SimpleNamespace(),1024,13)
    with patch('finetuning.build_80k.split_group',return_value='train'), \
         patch('finetuning.build_80k.ArbiterModel._prepare_decision',return_value={}):
        sel.add([row('a','English'),row('a','French')],['k','k'],whole=True)
    assert not sel.rows['train']


def test_norm_path_heldout_resolution():
    from pathlib import Path
    from finetuning.build_80k import norm_path
    repo_root = Path(__file__).resolve().parents[1]
    rel = Path('benchmark/data/cases.jsonl')
    abs_p = (repo_root / rel).resolve()
    assert norm_path(rel) == 'benchmark/data/cases.jsonl'
    assert norm_path(abs_p) == 'benchmark/data/cases.jsonl'
    outside = Path('/tmp/outside_file.jsonl')
    assert norm_path(outside) == '/tmp/outside_file.jsonl'



def test_verified_wrapper_rejects_recipe_quota_change(tmp_path):
    from scripts.prepare_adaptation import verify
    (tmp_path/'manifest.json').write_text(json.dumps({'recipe':'typed-boost-80k-v1','counts':{}}))
    import pytest
    with pytest.raises(ValueError, match='quotas'):
        verify(tmp_path, 'mixture_80k', 13)
