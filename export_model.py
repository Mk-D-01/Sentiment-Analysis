# Run locally (needs scikit-learn==1.8.0) whenever the model is retrained:
#   python export_model.py
# Converts the pickled CountVectorizer + LogisticRegression into model_weights.json,
# so the deployed site can predict with plain Python (no scikit-learn on Vercel).
import json
import pickle

with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vec = pickle.load(f)

params = vec.get_params()
assert params['ngram_range'] == (1, 1) and params['lowercase'] and not params['binary'], \
    "analysis.py only reproduces a lowercase unigram count vectorizer"

# One row of class weights per word: {"good": [w_neg, w_neu, w_pos], ...}
weights = {word: [round(float(c), 6) for c in model.coef_[:, idx]]
           for word, idx in vec.vocabulary_.items()}

out = {
    "token_pattern": params['token_pattern'],
    "classes": [int(c) for c in model.classes_],
    "intercept": [round(float(b), 6) for b in model.intercept_],
    "weights": weights,
}

with open('model_weights.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, separators=(',', ':'))

print(f"Exported {len(weights)} words to model_weights.json")
