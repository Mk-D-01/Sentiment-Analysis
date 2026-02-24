

import pickle

# Load them back
with open('sentiment_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    loaded_vec = pickle.load(f)

# Use them together

def predict(statement):
    new_sentence = [statement]
    transformed_input = loaded_vec.transform(new_sentence)
    prediction = loaded_model.predict(transformed_input)
    sentiment= ['Negative',"Neutral","Positive"]
    ind =prediction[0]
    return sentiment[ind]
    #return sentiment[prediction]

