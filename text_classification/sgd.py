import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


def sgd(X_train, y_train, X_test, y_test) -> Pipeline:
    """Train Linear Support Vector machine model"""
    sgd = Pipeline([('vect', CountVectorizer(lowercase=False)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier()),
                    ])
    sgd.fit(X_train, y_train)

    y_pred: Pipeline = sgd.predict(X_test)

    return y_pred


def score_sgd(csv_filename: str) -> float:
    """Выводит точность модели SGD по обучению
    на датасете"""
    df = pd.read_csv(csv_filename)
    X = df.review
    y = df.sentiment
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    y_pred: Pipeline = sgd(X_train, y_train, X_test, y_test)

    return accuracy_score(y_pred, y_test)


if __name__ == "__main__":
    accuracy = score_sgd("preprocessed_rureviews.csv")
    print(accuracy)
    # 0.7263
