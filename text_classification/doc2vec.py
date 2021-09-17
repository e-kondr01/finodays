import pandas as pd
import numpy as np

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import utils


def label_sentences(corpus, label_type):
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(TaggedDocument(v.split(), [label]))
    return labeled


def get_vectors(model, corpus_size, vectors_size, vectors_type):
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        prefix = vectors_type + '_' + str(i)
        vectors[i] = model.dv[prefix]
    return vectors


def doc2vec_logreg(X_train, y_train, X_test, y_test) -> Pipeline:
    """Train Logistic Regression model with doc2vec"""
    X_train = label_sentences(X_train, 'Train')
    X_test = label_sentences(X_test, 'Test')
    all_data = X_train + X_test

    model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5,
                         min_count=1, alpha=0.065, min_alpha=0.065)
    model_dbow.build_vocab(all_data)

    for epoch in range(30):
        model_dbow.train(utils.shuffle(
            all_data), total_examples=len(all_data),
            epochs=1)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha

    train_vectors_dbow = get_vectors(model_dbow, len(X_train), 300, 'Train')
    test_vectors_dbow = get_vectors(model_dbow, len(X_test), 300, 'Test')
    logreg = LogisticRegression(max_iter=400)
    logreg = logreg.fit(train_vectors_dbow, y_train)
    y_pred = logreg.predict(test_vectors_dbow)
    return y_pred


def score_doc2vec(csv_filename: str) -> float:
    """Выводит точность модели Logistic Regression с doc2vec по обучению
    на датасете"""
    df = pd.read_csv(csv_filename)
    X = df.review
    y = df.sentiment
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3)

    y_pred: Pipeline = doc2vec_logreg(X_train, y_train, X_test, y_test)

    return accuracy_score(y_pred, y_test)


if __name__ == "__main__":
    accuracy = score_doc2vec("preprocessed_rureviews.csv")
    print(accuracy)
    # 0.7184
