import pytest
from benchmark.targeted_report import metrics


def row(suite, gold=1, pred=1):
    return {'suite': suite, 'expected_answers': 1, 'observations': {
        'q': {'gold': gold, 'pred': pred, 'correct': gold == pred, 'kind': 'noul'}}}


def test_binary_micro_f1_and_macro():
    rows = [row('typed_decisions'), row('xnli.en'), row('fast_single.a'),
            row('fast_single.b',1,0), row('fast_multilabel.a'), row('fast_multilabel.a',0,1),
            row('fast_multilabel.a',1,0)]
    result = metrics(rows)
    assert result['Fast Single · macro'] == .5
    assert result['Fast Multi · micro F1'] == .5
    assert result['Fast Multi · exact match'] == pytest.approx(1/3)
    rows[-1]['error'] = 'failed'
    with pytest.raises(ValueError, match='failed'):
        metrics(rows)


def test_frozen_fixture_has_all_six_metrics():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / 'data'
    rows = []
    for path in [root/'cases.jsonl', root/'public_requested_v1/cases.jsonl']:
        for line in path.open():
            r = json.loads(line)
            if r['suite'] == 'typed_decisions' or r['suite'].startswith(('xnli.', 'fast_single.', 'fast_multilabel.')):
                rows.append({'suite':r['suite'], 'expected_answers':len(r['gold']),
                             'observations':{qid:{'gold':g['idx'],'pred':g['idx'],'correct':True,
                                                  'kind':r['questions'][qid]['type']}
                                             for qid,g in r['gold'].items()}})
    assert set(metrics(rows).values()) == {1.0}
