import pandas as pd
from feature_engineering import fit_vectorizer, save_vectorizer, custom_features
import numpy as np
from scipy.sparse import hstack

# Load your cleaned dataset
df = pd.read_csv("data/email_dataset.csv")

# Fit TF-IDF vectorizer
vectorizer, X_tfidf = fit_vectorizer(df['text'], max_features=3000)
save_vectorizer(vectorizer, "data/tfidf_vectorizer.pkl")

# Extract custom features
X_custom = custom_features(df['text'])

# Combine features
X_all = hstack([X_tfidf, X_custom])

# Save combined features and labels
np.save("data/X_all.npy", X_all.toarray())
df['label'].to_csv("data/y_labels.csv", index=False)
print(f"Feature matrix shape: {X_all.shape}")