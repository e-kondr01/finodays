import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text
from keras.utils import np_utils


df = pd.read_csv("preprocessed_rureviews.csv")
df = shuffle(df)

train_size = int(len(df) * .7)
train_posts = df["review"][:train_size]
train_tags = df["sentiment"][:train_size]

test_posts = df["review"][train_size:]
test_tags = df["sentiment"][train_size:]

max_words = 1000
tokenize = text.Tokenizer(
    num_words=max_words,
    lower=False
)
tokenize.fit_on_texts(train_posts)

x_train = tokenize.texts_to_matrix(
    train_posts,
    mode="tfidf"
)
x_test = tokenize.texts_to_matrix(
    test_posts,
    mode="tfidf"
)

encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

num_classes = np.max(y_train) + 1
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

batch_size = 32
epochs = 30

model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )


history = model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1
                    )

score = model.evaluate(x_test,
                       y_test,
                       batch_size=batch_size,
                       verbose=1
                       )
print(score[1])

# 0.7307
