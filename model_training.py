import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

def train_naive_bayes(X_train, y_train, alpha=1.0):
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train, C=1.0):
    model = LinearSVC(C=C, max_iter=2000)
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train, C=1.0):
    model = LogisticRegression(C=C, max_iter=2000, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)