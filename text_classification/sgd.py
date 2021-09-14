from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


def sgd(X_train, y_train, X_test, y_test):
    """Оценивание модели Linear Support Vector Machine"""
    sgd = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                          alpha=1e-3, random_state=42,
                                          max_iter=5, tol=None)),
                    ])
    sgd.fit(X_train, y_train)

    y_pred = sgd.predict(X_test)

    return accuracy_score(y_pred, y_test)
