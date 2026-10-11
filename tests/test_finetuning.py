"""Fine-tuning loss, label mapping and split isolation, no GPU or base weights."""
import json
import pytest
import torch
from finetuning.data import audit, read_rows
from finetuning.train import loss_terms, atomic_checkpoint


def row(id='a', group='a', context='Unique context'):
    return dict(id=id, group_id=group, family='policy', source='fixture', source_revision='v1', source_split='train',
                license='test', context=context, instruction='Choose.', question_type='choice',
                criteria={'a':'A','b':'B'}, target={'a':.25,'b':.75})


def test_contract_and_overlap(tmp_path):
    train, val = [row()], [row('b','b','Other context')]
    path = tmp_path/'heldout.jsonl'
    path.write_text(json.dumps({'state':'Reserved test'})+'\n')
    assert audit(train,val,[path])['train_rows'] == 1
    with pytest.raises(ValueError, match='overlap'):
        audit(train,[row('b','a','Other context')],[path])
    with pytest.raises(ValueError, match='overlap'):
        audit(train,[row('b','b',' UNIQUE   CONTEXT ')],[path])
    path.write_text(json.dumps({'state':'unique context'})+'\n')
    with pytest.raises(ValueError, match='Heldout'):
        audit(train,val,[path])
    data = tmp_path/'rows.jsonl'
    invalid = row(); invalid['target']['a'] = -.1
    data.write_text(json.dumps(invalid)+'\n')
    with pytest.raises(ValueError, match='probability'):
        read_rows(data)


def test_accumulation_gradient_matches_full_batch():
    torch.manual_seed(13)
    logits = torch.randn(5,3, requires_grad=True)
    target = torch.tensor([[1.,0,0],[.2,.3,.5],[0,1.,0],[0,0,1.],[.1,.2,.7]])
    mask = torch.ones_like(target,dtype=torch.bool)
    kinds = ['choice','score','noul','choice','score']
    ce, mae, score = loss_terms(logits,target,mask,kinds)
    full = ce.mean()+.5*mae[score].mean()
    expected, = torch.autograd.grad(full,logits)
    split = logits.detach().clone().requires_grad_()
    for lo,hi in ((0,2),(2,4),(4,5)):
        c,m,s = loss_terms(split[lo:hi],target[lo:hi],mask[lo:hi],kinds[lo:hi])
        (c.sum()/5+.5*m[s].sum()/2).backward()
    torch.testing.assert_close(expected,split.grad)


def test_padding_mask_and_score_values():
    logits = torch.tensor([[0.,0.,999.]], requires_grad=True)
    target = torch.tensor([[0.,1.,0.]])
    ce,mae,_ = loss_terms(logits,target,torch.tensor([[True,True,False]]),['score'])
    assert ce.item() == pytest.approx(.69314718)
    assert mae.item() == pytest.approx(.5)
    ce.sum().backward()
    assert logits.grad[0,2] == 0


def test_atomic_checkpoint_rng_roundtrip(tmp_path):
    path = tmp_path/'latest.pt'
    atomic_checkpoint(path, {'step': 1, 'rng':torch.get_rng_state()})
    saved = torch.load(path, weights_only=True)
    assert saved['step'] == 1
    assert torch.equal(saved['rng'], torch.get_rng_state())
    assert not path.with_suffix('.tmp').exists()
