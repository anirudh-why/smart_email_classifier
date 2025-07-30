import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from model_training import (
    train_naive_bayes, train_logistic_regression, evaluate_model, save_model
)

# Load features and labels
X = np.load("data/X_all.npy")
y = pd.read_csv("data/y_labels.csv")["label"].values

# Split train/test (stratify to preserve class proportions)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Balance the training set
ros = RandomOverSampler(random_state=42)
X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)
print(f"Original training set: {X_train.shape}, Balanced: {X_train_bal.shape}")

# Train Naive Bayes
nb_model = train_naive_bayes(X_train_bal, y_train_bal)
print("== Naive Bayes ==")
evaluate_model(nb_model, X_test, y_test)
save_model(nb_model, "data/spam_classifier_nb_balanced.pkl")

# Train Logistic Regression (often superior for text)
lr_model = train_logistic_regression(X_train_bal, y_train_bal)
print("== Logistic Regression ==")
evaluate_model(lr_model, X_test, y_test)
save_model(lr_model, "data/spam_classifier_lr_balanced.pkl")