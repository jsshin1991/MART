from table.proto_critic import kernel, greedy_select_protos, select_criticism_regularized
import numpy as np
import pandas as pd
import gower
from tqdm import tqdm

def relative_proximity(input, proto_critics_list, kernel_function):
    """Computes relative proximity of input relative to the original data

    Args:
      input: input_data
      proto_critics_list: proto_critics list in the original data
      kernel_function: kernel function

    Returns:
      rel_prox: relative proxity for the input
    """

    rel_prox_list = kernel_function(proto_critics_list, input)
    # rel_prox = np.amax(rel_prox_list, axis=1)
    rel_prox = np.amax(rel_prox_list)

    return rel_prox

def kernel_function(proto_critic, input):
    gamma = 1
    kernel = np.array([])
    for idx in range(input.shape[0]):
        dist = gower.gower_matrix(data_x=proto_critic, data_y=np.reshape(input[idx, :], (1, -1))).reshape((-1))
        if idx == 0:
            kernel = np.exp(-gamma * dist)
        else:
            kernel = np.vstack((kernel, np.exp(-gamma * dist)))
    return kernel

def run_glocal_MART(input, X, model_function, proto_critics_idx, kernel_function, classification, weight_function = None, steps = 100):
    """Computes GLocal MART for a given target instance, model function.

    Args:
      input: The specific inputs for which GLocal Inference must be computed.
      X: Whole dataset
      model_function: model function.
      proto_critics_idx: proto-critics index for X.
      weight_function: weight function. Default = None.
      kernel_function: kernel function to compute relative proximities.
      classification: whether the prediction is classification or not.
      steps: [optional] These steps along determine the integral approximation error. By default, steps is set to 100.

    Returns:
      glm: Output for GLocal Mart.
    """

    c_i = np.zeros(shape=(input.shape[0], input.shape[1]))
    min = X.min()
    max = X.max()

    proto_critics_list = np.array(X.iloc[proto_critics_idx, :])

    for i in tqdm(range(input.shape[1])):
        if max[i] == min[i]:
            c_i[:, i] = 0
            continue
        e = (min[i]-input[:, i])/(max[i]-min[i])
        for j in range(steps + 1):
            w = np.abs(1/e) # temp weight_function
            x = input.copy()
            x[:, i] += (max[i]-min[i])*e
            drx = relative_proximity(input=x, proto_critics_list=proto_critics_list, kernel_function=kernel_function)
            if classification:
                diff = np.abs(model_function.predict_proba(x) - model_function.predict_proba(input))
                diff = np.average(diff, axis=1)
            else:
                diff = np.abs(model_function.predict(x) - model_function.predict(input))

            c_i[:, i] += np.multiply(np.multiply(w, drx), diff)
            e += 1 / steps

    glm = np.zeros_like(c_i)
    c_i_sum = np.sum(c_i, axis=1)
    for idx in range(glm.shape[0]):
        glm[idx] = c_i[idx] / c_i_sum[idx]
    return glm
