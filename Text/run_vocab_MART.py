import numpy as np
from Text.SimpleRNNModel.rnn_utils import word2vec, vec2classifier
from tqdm import tqdm


def weight_function(x):
    # return np.sum(np.abs(x), axis=1)
    return min(1 / (np.sum(np.abs(x), axis=1) + 1e-12), 1e12)


# input: 2D np array whose elements are indices for words
def run_vocab_MART(input, model, vocab_size, idx_list, steps=30):
    # word to vector model
    word2vecModel = word2vec(model, vocab_size)
    # vector to classifier model
    vec2classifierModel = vec2classifier(model)
    # whole vector
    whole_vec = word2vecModel.predict(list(idx_list))
    whole_vec = whole_vec.reshape(whole_vec.shape[0], whole_vec.shape[2])
    vec_min = np.min(whole_vec, axis=0)
    vec_max = np.max(whole_vec, axis=0)
    # input vector
    input_vec = word2vecModel.predict(input)
    # null vector
    null_vec = word2vecModel.predict([0])
    null_vec = null_vec.reshape(null_vec.shape[2])

    half_steps = int(steps / 2)
    input_pred = vec2classifierModel.predict(input_vec)
    c_i = np.zeros(shape=(input_vec.shape[0], input_vec.shape[1]))

    for i in tqdm(range(input_vec.shape[1])):
        null_check = np.sum(input_vec[:, i] == null_vec, axis=1)
        null_check = null_check.reshape(null_check.shape[0]) != input_vec.shape[2]

        e = (vec_min - input_vec[:, i])
        for it in range(half_steps + 1):
            w = weight_function(e)
            x = input_vec.copy()
            x[:, i] += e

            diff = np.abs(vec2classifierModel.predict(x) - input_pred)
            diff = diff.reshape(diff.shape[0])
            c_i[:, i] += w * diff * null_check
            e += (input_vec[:, i] - vec_min) / half_steps

        e = (vec_max - input_vec[:, i])
        for it in range(half_steps + 1):
            w = weight_function(e)
            x = input_vec.copy()
            x[:, i] += e

            diff = np.abs(vec2classifierModel.predict(x) - input_pred)
            diff = diff.reshape(diff.shape[0])
            c_i[:, i] += w * diff * null_check
            e += (input_vec[:, i] - vec_max) / half_steps

    mart = np.zeros_like(c_i)
    c_i_sum = np.sum(c_i, axis=1)
    for idx in range(mart.shape[0]):
        mart[idx] = c_i[idx] / c_i_sum[idx]

    return mart
