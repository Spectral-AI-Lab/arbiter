"""Probability-based metrics; never use the model-specific confidence field."""
import math
import numpy as np


def observation(question, answer, gold):
    kind = question['type']
    if kind == 'noul':
        p = float(answer['noul'])
        probabilities = np.array([1-p, p])
    else:
        keys = list(question['criteria']) if kind == 'choice' else [str(i) for i in range(len(question['criteria']))]
        probabilities = np.array([answer['probabilities'][k] for k in keys], dtype=float)
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1, atol=2e-3):
        raise ValueError('Invalid probability distribution')
    probabilities = probabilities / probabilities.sum()  # Laya may round probabilities.
    expected = int(gold['idx'])
    if not 0 <= expected < len(probabilities):
        raise ValueError('Gold label is outside candidate set')
    pred = int(probabilities.argmax())
    result = {'kind': kind, 'gold': expected, 'pred': pred,
              'correct': pred == expected, 'pmax': float(probabilities.max()),
              'brier': float(np.sum((probabilities - np.eye(len(probabilities))[expected])**2)),
              'nll': -math.log(max(float(probabilities[expected]), 1e-12)),
              'probabilities': probabilities.tolist()}
    if 'gold_score' in gold:
        result['score_error'] = abs(float(np.arange(len(probabilities)) @ probabilities) - gold['gold_score'])
    if 'soft' in gold and sum(gold['soft']) > 0:
        soft = np.asarray(gold['soft'], float)
        soft /= soft.sum()
        result['soft_accuracy'] = float(probabilities @ soft)
        result['brier_vs_soft'] = float(np.sum((probabilities-soft)**2))
    return result


def summarize(observations):
    if not observations:
        return {'n': 0}
    conf = np.array([r['pmax'] for r in observations])
    correct = np.array([r['correct'] for r in observations], float)
    ece = 0.0
    for i in range(15):
        mask = ((conf >= i/15) if i == 0 else (conf > i/15)) & (conf <= (i+1)/15)
        if mask.any():
            ece += mask.mean() * abs(conf[mask].mean()-correct[mask].mean())
    out = {'n': len(observations), 'accuracy': float(correct.mean()), 'ece_15': float(ece),
           'mean_pmax': float(conf.mean())}
    for name in ['brier', 'nll', 'soft_accuracy', 'brier_vs_soft', 'score_error']:
        values = [r[name] for r in observations if name in r]
        if values:
            out['score_mae' if name == 'score_error' else name] = float(np.mean(values))
    # Threshold selection keeps equal-confidence ties together.
    cutoff = np.sort(conf)[::-1][max(0, math.ceil(len(conf)*.5)-1)]
    keep = conf >= cutoff
    out.update(selective_accuracy_50=float(correct[keep].mean()), actual_coverage_50=float(keep.mean()))
    return out
