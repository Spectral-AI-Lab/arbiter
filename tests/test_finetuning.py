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


def test_source_grouping_and_soft_labels():
    from finetuning.prepare_sources import split_group, typed_rows
    source = dict(id='train-1', workflow='invoice', state='{"n":1}',
                  questions={'severity': {'type':'score','instructions':'Rate.', 'criteria':{'0':'low','1':'high'}}},
                  gold={'severity': {'probabilities':{'0':.3,'1':.7}}})
    r, = list(typed_rows(source, 'a'*40))
    assert r['target'] == {'0':.3,'1':.7}
    assert r['group_id'] == 'typed:train-1'
    assert split_group(r['group_id'],13) == split_group(r['group_id'],13)


def test_training_resume_matches_uninterrupted(tmp_path, monkeypatch):
    """Real optimizer/backward/scheduler/RNG with a tiny CPU fixture model."""
    import sys
    from types import SimpleNamespace
    import finetuning.train as trainer
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = torch.nn.Sequential(torch.nn.Embedding(8,4), torch.nn.Dropout(.1))
            self.decision_head = torch.nn.Linear(4,2)
            self.tokenizer = SimpleNamespace(pad_token_id=0)
        def _prepare_decision(self, **r):
            return dict(ids=[1,2], positions=[0,1], keys=list(r['criteria']), question_type=r['question_type'])
        def forward(self, input_ids, **kwargs):
            logits = self.decision_head(self.language_model(input_ids).mean(1))
            return {'logits': logits, 'mask': torch.ones_like(logits,dtype=torch.bool)}
    monkeypatch.setattr(trainer.ArbiterModel,'from_pretrained',lambda *args,**kw: Tiny())
    monkeypatch.setattr(trainer,'export',lambda model,path: None)
    base = tmp_path/'base';base.mkdir()
    tr=tmp_path/'train.jsonl'; va=tmp_path/'val.jsonl'; held=tmp_path/'heldout.jsonl'
    tr.write_text('\n'.join(json.dumps(row(str(i),str(i),f'train {i}')) for i in range(5)))
    va.write_text(json.dumps(row('v','v','validation')))
    held.write_text(json.dumps({'state':'reserved'}))
    def run(output,resume=False):
        args = ['train','--model',str(base),'--train',str(tr),'--validation',str(va),
                '--heldout',str(held),'--output',str(output),'--device','cpu','--epochs','2',
                '--batch-size','2','--accumulation','2','--eval-every','1']
        if resume: args += ['--resume']
        monkeypatch.setattr(sys,'argv',args)
        trainer.main()
    full, interrupted = tmp_path/'full', tmp_path/'interrupted'
    run(full)
    save = trainer.atomic_checkpoint
    def stop_after_save(path,payload):
        save(path,payload)
        raise InterruptedError('simulated stop after complete optimizer step')
    monkeypatch.setattr(trainer,'atomic_checkpoint',stop_after_save)
    with pytest.raises(InterruptedError): run(interrupted)
    monkeypatch.setattr(trainer,'atomic_checkpoint',save)
    run(interrupted,True)
    a,b = [torch.load(p/'latest.pt',weights_only=False) for p in (full,interrupted)]
    assert a['step'] == b['step'] == 4
    for key in a['model']:
        torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
