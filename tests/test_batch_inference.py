"""CPU integration checks for real batched inference and scheduling budgets."""
import pytest

pytestmark = pytest.mark.integration

from pathlib import Path
from unittest.mock import patch
import torch
from arbiter import ArbiterModel

@pytest.fixture(scope='module')
def model():
    return ArbiterModel.from_pretrained(Path(__file__).resolve().parents[1] / 'checkpoints/arbiter-272m', device='cpu')

@pytest.fixture
def requests():
    return [
        dict(context='The parcel is delayed. ' * 8, instruction='Classify the issue.', criteria=['delay', 'payment', 'fraud']),
        dict(context='Earth orbits the sun.', instruction='Is this true?', criteria=None, question_type='noul'),
        dict(context='All services are unavailable.', instruction='Assess severity.', criteria=['low', 'medium', 'high', 'critical'], question_type='score'),
    ]

def test_mixed_batch_parity_and_one_forward(model, requests):
    singles = [model.decide(**r) for r in requests]
    with patch.object(model, 'forward', wraps=model.forward) as forward:
        batch = model.decide_batch(requests)
        assert forward.call_count == 1
        assert forward.call_args.kwargs['input_ids'].shape[0] == 3
    for single, result in zip(singles, batch):
        assert single['selected_key'] == result['selected_key']
        assert single['input_tokens'] == result['input_tokens']
        torch.testing.assert_close(torch.tensor(list(single['probabilities'].values())),
                                   torch.tensor(list(result['probabilities'].values())), atol=.02, rtol=0)
    assert batch[1]['noul'] == batch[1]['probabilities']['true']
    assert batch[2]['score'] == sum(i*p for i,p in enumerate(batch[2]['probabilities'].values()))

def test_budgets_and_validation_before_forward(model, requests):
    lengths = [len(model._prepare_decision(**r)['ids']) for r in requests]
    limit = max(lengths)
    with patch.object(model, 'forward', wraps=model.forward) as forward:
        model.decide_batch(requests, max_batch_tokens=limit, max_attention_elements=limit**2)
        for call in forward.call_args_list:
            b,l = call.kwargs['input_ids'].shape
            assert b*l <= limit and b*l*l <= limit**2
    with patch.object(model, 'forward', wraps=model.forward) as forward:
        with pytest.raises(ValueError, match='Request 3'):
            model.decide_batch(requests + [dict(requests[0], context='')], max_batch_tokens=1_000_000, max_attention_elements=16777216)
        assert forward.call_count == 0


def test_empty_singleton_and_predict_mapping(model, requests):
    assert model.decide_batch([]) == []
    assert model.predict_batch([]) == []
    assert model.predict('x', {}) == {'answers': {}}
    inputs = [dict(state=r['context'], questions={str(i): dict(instructions=r['instruction'], criteria=r['criteria'], type=r.get('question_type','choice'))}) for i,r in enumerate(requests)]
    with patch.object(model, 'forward', wraps=model.forward) as forward:
        outputs = model.predict_batch(inputs)
        assert forward.call_count == 1
    assert [list(o['answers']) for o in outputs] == [['0'], ['1'], ['2']]
    assert 'noul' in outputs[1]['answers']['1']
    assert 'score' in outputs[2]['answers']['2']
    assert model.decide_batch(requests[:1])[0] == model.decide(**requests[0])


def test_truncation_and_invalid_limits(model, requests):
    with patch.object(model, 'forward', wraps=model.forward) as forward:
        result = model.decide_batch([dict(requests[0], context='word '*500)], max_length=64, allow_truncation=True)
        assert result[0]['truncated']
        assert forward.call_args.kwargs['input_ids'].shape[1] <= 64
    for key in ('batch_size', 'max_batch_tokens', 'max_attention_elements', 'max_length'):
        with pytest.raises(ValueError):
            model.decide_batch(requests, **{key: 0})


def test_singleton_budget_and_eval_guard(model, requests):
    with patch.object(model, 'forward', wraps=model.forward) as forward:
        with pytest.raises(ValueError, match='singleton'):
            model.decide_batch(requests, max_batch_tokens=1)
        assert forward.call_count == 0
    model.train()
    try:
        with pytest.raises(RuntimeError, match='eval'):
            model.decide_batch(requests)
    finally:
        model.eval()


def test_reserved_label_rejected_without_changing_other_keys(model, requests):
    with patch.object(model, 'forward', wraps=model.forward) as forward:
        with pytest.raises(ValueError, match='reserved'):
            model.decide(**dict(requests[0],criteria={'<OPTION>':'a','[OPTION]':'b'}))
        assert forward.call_count == 0
    result=model.decide(**dict(requests[0],criteria={'[OPTION]':'literal <OPTION>','other':'b'}))
    assert list(result['probabilities']) == ['[OPTION]','other']


def test_single_encoding_for_nontruncated_input(model, requests):
    with patch.object(model.tokenizer,'encode',wraps=model.tokenizer.encode) as encode:
        model._prepare_decision(**requests[0])
        assert encode.call_count == 1


def test_structured_state_instruction_alias_and_conflicts(model):
    import json
    state=[{'role':'user','content':'Échec de livraison'}]
    question={'instruction':'Classify','criteria':['delivery','payment']}
    with patch.object(model,'decide_batch',return_value=[{'probabilities':{'0':.8,'1':.2},'selected_key':'0'}]) as batch:
        model.predict(state,{'q':question})
        assert batch.call_args.args[0][0]['context'] == json.dumps(state,ensure_ascii=False)
        assert batch.call_args.args[0][0]['instruction'] == 'Classify'
    with pytest.raises(ValueError,match='conflicting'):
        model.predict(state,{'q':dict(question,instructions='Different')})
    result=model.predict(state,{'q':dict(question,instructions='Classify')})
    assert 'q' in result['answers']


def test_result_tie_and_ordinal_index_contract():
    item=dict(keys=['1','2','3'],question_type='score',raw_token_count=10,is_truncated=False)
    result=ArbiterModel._decision_result(item,[.5,.5,0.0])
    assert result['selected_key']=='1'
    assert result['score']==.5


@pytest.mark.parametrize('path',['missing/remote','checkpoints/arbiter-272m'])
def test_loader_rejects_unsupported_keywords_before_loading(path):
    with pytest.raises(TypeError,match='Unsupported loading arguments'):
        ArbiterModel.from_pretrained(path,unsupported_flag='invalid')


def test_loader_accepts_device_map_auto():
    path = Path(__file__).resolve().parents[1] / 'checkpoints/arbiter-272m'
    model = ArbiterModel.from_pretrained(path, device_map='auto', local_files_only=True)
    assert model is not None
    assert isinstance(model, ArbiterModel)


def test_loader_rejects_hub_options_on_local_path():
    path=Path(__file__).resolve().parents[1]/'checkpoints/arbiter-272m'
    with pytest.raises(TypeError,match='only to remote'):
        ArbiterModel.from_pretrained(path,revision='main')
