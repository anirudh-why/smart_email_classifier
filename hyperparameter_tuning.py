import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
import pickle

X = np.load("data/X_all.npy")
y = pd.read_csv("data/y_labels.csv")["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

ros = RandomOverSampler(random_state=42)
X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)

# Naive Bayes
nb_params = {'alpha': [0.1, 0.5, 1.0]}
nb_grid = GridSearchCV(MultinomialNB(), nb_params, cv=3, scoring='f1_macro')
nb_grid.fit(X_train_bal, y_train_bal)
print("Best NB params:", nb_grid.best_params_)
print("NB test set report:")
print(classification_report(y_test, nb_grid.best_estimator_.predict(X_test)))
pickle.dump(nb_grid.best_estimator_, open("data/spam_classifier_nb_tuned.pkl", "wb"))

# Logistic Regression with faster solver and smaller grid
lr_params = {'C': [0.1, 1, 10]}
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=500, class_weight='balanced', solver='liblinear'),
    lr_params, cv=3, scoring='f1_macro'
)
lr_grid.fit(X_train_bal, y_train_bal)
print("Best LR params:", lr_grid.best_params_)
print("LR test set report:")
print(classification_report(y_test, lr_grid.best_estimator_.predict(X_test)))
pickle.dump(lr_grid.best_estimator_, open("data/spam_classifier_lr_tuned.pkl", "wb"))