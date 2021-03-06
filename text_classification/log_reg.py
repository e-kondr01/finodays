import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def log_reg(X_train, y_train, X_test, y_test) -> Pipeline:
    """Train Logistic Regression model"""
    logreg = Pipeline([('vect', CountVectorizer(lowercase=False)),
                       ('tfidf', TfidfTransformer()),
                       ('clf', LogisticRegression(max_iter=400)),
                       ])
    logreg.fit(X_train, y_train)

    y_pred: Pipeline = logreg.predict(X_test)

    return y_pred


def score_log_reg(csv_filename: str) -> float:
    """Выводит точность модели Logistic Regression по обучению
    на датасете"""
    df = pd.read_csv(csv_filename)
    X = df.review
    y = df.sentiment
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3)

    y_pred: Pipeline = log_reg(X_train, y_train, X_test, y_test)

    return accuracy_score(y_pred, y_test)


def save_log_reg():
    df = pd.read_csv("preprocessed_rureviews.csv")
    X = df.review
    y = df.sentiment
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3)
    logreg = Pipeline([('vect', CountVectorizer(lowercase=False)),
                       ('tfidf', TfidfTransformer()),
                       ('clf', LogisticRegression(max_iter=400)),
                       ])
    logreg = logreg.fit(X_train, y_train)
    joblib.dump(logreg, "./logreg.joblib")


if __name__ == "__main__":
    accuracy = score_log_reg("preprocessed_rureviews.csv")
    print(accuracy)
    #  0.7317
