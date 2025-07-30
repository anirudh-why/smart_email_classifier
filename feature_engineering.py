from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
import pickle

def count_urls(text):
    return len(re.findall(r'(https?://\S+)', text))

def count_exclamations(text):
    return text.count('!')

def capital_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0
    capitals = [c for c in letters if c.isupper()]
    return len(capitals) / len(letters)

def email_length(text):
    return len(text)

def custom_features(corpus):
    feats = []
    for text in corpus:
        feats.append([
            count_urls(text),
            count_exclamations(text),
            capital_ratio(text),
            email_length(text)
        ])
    return np.array(feats)

def fit_vectorizer(corpus, max_features=3000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X

def transform_text(vectorizer, corpus):
    return vectorizer.transform(corpus)

def save_vectorizer(vectorizer, path):
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)

def load_vectorizer(path):
    with open(path, "rb") as f:
        return pickle.load(f)