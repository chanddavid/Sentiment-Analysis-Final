# model/predict.py
import sys
import numpy as np
from tensorflow.keras.models import load_model
from model.data_cleaning import clean_text, preprocess_text

# Load model once (global for efficiency)
_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_model('../sentiment_model.keras')
    return _model

def predict_sentiment(text):
    """Return (sentiment, confidence) for a raw text review."""
    model = get_model()
    cleaned = clean_text(text)
    processed = preprocess_text(cleaned)
    prob = model.predict([processed], verbose=0)[0][0]
    if prob >= 0.5:
        return "Positive", float(prob * 100)
    else:
        return "Negative", float((1 - prob) * 100)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        review = " ".join(sys.argv[1:])
    else:
        review = input("Enter a product review: ")
    sentiment, confidence = predict_sentiment(review)
    print(f"Sentiment: {sentiment} (confidence: {confidence:.2f}%)")