from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from db.data import get_train_data, get_test_data
from text_classification.preprocess import preprocess_text


def bayes():
    data_train = map(lambda it: (it[0], it[1]), get_train_data())
    mood_train: list[int] = []
    text_train: list[str] = []
    j = 0
    for i in data_train:
        mood_train.append(i[0])
        text_train.append(" ".join(preprocess_text(i[1])))
        j += 1
        if j == 5:
            break

    model = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
                      ])
    model.fit(text_train, mood_train)

    data_test = map(lambda it: (it[0], it[1]), get_test_data())
    mood_test: list[int] = []
    text_test: list[str] = []
    j = 0
    for i in data_test:
        mood_test.append(i[0])
        text_test.append(" ".join(preprocess_text(i[1])))
        j += 1
        if j==5:
            break

    print(model.score(text_test, mood_test))


if __name__ == "__main__":
    bayes()
