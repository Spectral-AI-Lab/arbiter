"""Adapter from GLiNER2.5-Decide classification to the benchmark contract."""
import json


class GLiNERDecisionAgent:
    def __init__(self, repo, device):
        eager = False
        try:
            from gliner2 import AutoExtractor
        except ImportError:  # gliner2 releases before the AutoExtractor rename
            from gliner2 import GLiNER2 as AutoExtractor
            eager = True
            class EagerGLiNER(AutoExtractor):
                def __init__(self, config, *args, **kwargs):
                    config._attn_implementation = 'eager'
                    config._attn_implementation_internal = 'eager'
                    super().__init__(config, *args, **kwargs)
            AutoExtractor = EagerGLiNER
        self.repo = repo
        # Older releases use ``map_location``; newer ones accept ``device``.
        # GLiNER2 >=2.0 selects the architecture from the checkpoint and
        # accepts only map_location. Older releases need the eager override.
        if eager:
            self.model = AutoExtractor.from_pretrained(repo, map_location=device)
        else:
            self.model = AutoExtractor.from_pretrained(repo, map_location=device)
        self.device = device

    @staticmethod
    def _state(state):
        return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True)

    def predict(self, state, questions):
        text = self._state(state)
        answers = {}
        for qid, q in questions.items():
            criteria = q.get('criteria')
            if q['type'] == 'score':
                labels = {str(i): label for i, label in enumerate(criteria)}
            elif q['type'] == 'noul':
                labels = criteria or {'false': 'The statement is false.', 'true': 'The statement is true.'}
            else:
                labels = criteria
            result = self.model.classify_text(text, {qid: labels}, include_confidence=True)
            value = result[qid]
            if isinstance(value, dict):
                label, confidence = value['label'], float(value.get('confidence', 1.0))
            else:
                label, confidence = value, 1.0
            confidence = min(max(confidence, 1e-6), 1.0)
            rest = (1.0 - confidence) / max(1, len(labels) - 1)
            probs = {str(k): rest for k in labels}
            probs[str(label)] = confidence
            if q['type'] == 'noul':
                answers[qid] = {'noul': probs['true']}
            elif q['type'] == 'score':
                answers[qid] = {'probabilities': probs}
            else:
                answers[qid] = {'probabilities': probs}
        return {'model': self.repo, 'answers': answers, 'usage': {}}
