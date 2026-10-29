import json
import math
import os
import re
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Weights exported from the pickled model by export_model.py
with open(os.path.join(BASE_DIR, 'model_weights.json'), encoding='utf-8') as f:
    _model = json.load(f)

_token_re = re.compile(_model['token_pattern'])
_weights = _model['weights']
_intercept = _model['intercept']
_classes = _model['classes']

sentiment = ['Negative', "Neutral", "Positive"]


def _scores(statement):
    # Same as CountVectorizer.transform + LogisticRegression.predict_proba
    counts = Counter(_token_re.findall(statement.lower()))
    logits = list(_intercept)
    known_words = 0
    for word, n in counts.items():
        w = _weights.get(word)
        if w is None:
            continue
        known_words += 1
        for k in range(len(logits)):
            logits[k] += n * w[k]
    top = max(logits)
    exps = [math.exp(z - top) for z in logits]
    total = sum(exps)
    return [e / total for e in exps], known_words


def predict(statement):
    probs, _ = _scores(statement)
    ind = _classes[probs.index(max(probs))]
    return sentiment[ind]


def predict_with_scores(statement):
    # Returns the label plus the probability of each class
    probs, known_words = _scores(statement)
    scores = {sentiment[c]: round(p, 4) for c, p in zip(_classes, probs)}
    label = max(scores, key=scores.get)
    # known_words = distinct words the model has seen; 0 means the result is just its default guess
    return {"sentiment": label, "scores": scores, "known_words": known_words}
