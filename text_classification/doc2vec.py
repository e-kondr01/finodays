import numpy as np
import re

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn import utils


def label_sentences(corpus, label_type):
    """
    Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
    We do this by using the TaggedDocument method. The format will be "TRAIN_i" or "TEST_i" where "i" is
    a dummy index of the post.
    """
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(TaggedDocument(v.split(), [label]))
    return labeled


def train_model(X_train, X_test):
    X_train = label_sentences(X_train, 'Train')
    X_test = label_sentences(X_test, 'Test')
    all_data = X_train + X_test

    model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5,
                         min_count=1, alpha=0.065, min_alpha=0.065)
    model_dbow.build_vocab([x for x in tqdm(all_data)])

    for epoch in range(30):
        model_dbow.train(utils.shuffle(
            [x for x in tqdm(all_data)]), total_examples=len(all_data), epochs=1)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha

    return model_dbow


def get_vectors(model, corpus_size, vectors_size, vectors_type):
    """
    Get vectors from trained doc2vec model
    :param doc2vec_model: Trained Doc2Vec model
    :param corpus_size: Size of the data
    :param vectors_size: Size of the embedding vectors
    :param vectors_type: Training or Testing vectors
    :return: list of vectors
    """
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        prefix = vectors_type + '_' + str(i)
        vectors[i] = model.docvecs[prefix]
    return vectors
