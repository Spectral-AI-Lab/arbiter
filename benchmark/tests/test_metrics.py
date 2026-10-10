import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import observation, summarize
from prepare import notebook_cells
from compare import compare


class MetricsTests(unittest.TestCase):
    def test_probabilities_not_reported_confidence(self):
        q={'type':'choice','criteria':{'a':'A','b':'B'}}
        o=observation(q, {'probabilities':{'b':.2,'a':.8},'confidence':0}, {'idx':0})
        self.assertTrue(o['correct'])
        self.assertAlmostEqual(o['brier'], .08)
        self.assertAlmostEqual(summarize([o])['ece_15'], .2)
        self.assertAlmostEqual(o['nll'], -np.log(.8))

    def test_ordinal_and_soft_targets(self):
        o=observation({'type':'score','criteria':['a','b','c']},
                      {'probabilities':{'0':.25,'1':.5,'2':.25}},
                      {'idx':1,'gold_score':1.5,'soft':[.25,.5,.25]})
        self.assertAlmostEqual(o['score_error'], .5)
        self.assertAlmostEqual(o['brier_vs_soft'], 0)
        self.assertAlmostEqual(o['soft_accuracy'], .375)

    def test_noul_and_ties(self):
        a=observation({'type':'noul'},{'noul':.8},{'idx':1})
        b=observation({'type':'noul'},{'noul':.8},{'idx':0})
        summary=summarize([a,b])
        self.assertEqual(summary['actual_coverage_50'],1)
        self.assertEqual(summary['selective_accuracy_50'],.5)
        with self.assertRaises(ValueError):
            observation({'type':'noul'},{'noul':1.2},{'idx':0})

    def test_upstream_builder_extraction(self):
        cells=notebook_cells()
        for marker in ['def build_choice_questions(', 'td_cases, td_golds, td_wf =',
                       'MASSIVE_LANGS =', '# SST-5: ordinal score']:
            matches=[c for c in cells if marker in c]
            self.assertEqual(len(matches),1)
            compile(matches[0],marker,'exec')

    def test_pairing_rejects_different_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name,sha in [('embeddinggemma2','a'),('laya','b')]:
                p=root/name;p.mkdir()
                (p/'summary.json').write_text('{}')
                (p/'predictions.jsonl').write_text('')
                (p/'metadata.json').write_text(json.dumps({'status':'completed','selection_sha256':sha,'dataset_sha256':'same','threads':8,'device':'cpu','batch_size':1}))
            with self.assertRaisesRegex(ValueError,'selection_sha256'):
                compare(root)


if __name__=='__main__':
    unittest.main()
