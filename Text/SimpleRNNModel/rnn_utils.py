from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import numpy as np


def train_RNNModel(X_train, y_train, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 32))
    model.add(SimpleRNN(32))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2)
    return model


def word2vec(model, vocab_size):
    word2vec = Sequential()
    word2vec.add(Embedding(vocab_size, 32))
    word2vec.set_weights(model.layers[0].get_weights())
    word2vec.trainable = False
    word2vec.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    return word2vec


def vec2classifier(model):
    vec2classifier = Sequential()
    vec2classifier.add(SimpleRNN(32, batch_input_shape=(None, None, 32)))
    vec2classifier.add(Dense(1, activation='sigmoid'))
    vec2classifier.layers[0].set_weights(model.layers[1].get_weights())
    vec2classifier.layers[1].set_weights(model.layers[2].get_weights())
    vec2classifier.trainable = False
    vec2classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    return vec2classifier


# input: one instance, mart: 1D np array inference for input
def show_rank(input, mart, index_to_word, threshold=1e-4):
    max_mart = np.max(mart)
    if max_mart > 1e-1:
        sorted_mart = np.sort(mart)
        max_mart = sorted_mart[-2]
    transf_mart = mart / max_mart
    sort = np.argsort(-transf_mart)

    imp_rank = []
    for idx in sort:
        if transf_mart[idx] >= threshold:
            imp_rank.append((idx, index_to_word[input[idx]]))
        else:
            break
    return imp_rank
