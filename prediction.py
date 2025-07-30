import numpy as np
from feature_engineering import load_vectorizer, transform_text, custom_features
from model_training import load_model
import re
from scipy.sparse import hstack

def clean_text(text):
    # Basic clean: lowercase, remove extra whitespace
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def predict_email(email_text, model_path="data/spam_classifier_lr_tuned.pkl", vectorizer_path="data/tfidf_vectorizer.pkl"):
    vectorizer = load_vectorizer(vectorizer_path)
    model = load_model(model_path)
    cleaned_text = clean_text(email_text)
    X_tfidf = transform_text(vectorizer, [cleaned_text])
    X_custom = custom_features([email_text])
    X_all = hstack([X_tfidf, X_custom])
    label = model.predict(X_all)[0]
    # Confidence
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_all)[0]
        confidence = np.max(proba)
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X_all)[0]
        # Logistic regression decision_function is log-odds, convert to probability
        confidence = 1 / (1 + np.exp(-decision))
    else:
        confidence = None
    return label, confidence