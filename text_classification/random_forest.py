from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

from db.data import get_train_data, get_test_data
from text_classification.preprocess import preprocess_text


def random_forest():
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

    cv = CountVectorizer(
        strip_accents='unicode',
        lowercase=True
    )
    matrix = cv.fit_transform(text_train)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(mood_train)
    model = RandomForestClassifier()
    model.fit(matrix, labels)

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

    cv = CountVectorizer(
        strip_accents='unicode',
        lowercase=True,
        vocabulary=cv.vocabulary_
    )
    matrix = cv.fit_transform(text_test)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(mood_test)

    print(model.score(matrix, labels))


if __name__ == "__main__":
    random_forest()
