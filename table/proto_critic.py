from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import gower
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
import sys
import math

# Create kernel matrix for mixed datatype
def kernel(table, gamma=1, dbtype = 'mixed'):
    if dbtype == 'mixed':
        dist = gower.gower_matrix(table)
        # kernel = np.power(table.shape[0], -gamma*dist)
        kernel = np.exp(-gamma*dist)
        # kernel = rbf_kernel(table, gamma)
        # kernel = np.power(dist, np.shape(table)[0])
    return kernel

def cost_function(K, selected, colsum, candidates, is_K_sparse=False):
    s1array = colsum[candidates]
    if len(selected) > 0:
        temp = K[selected, :][:, candidates]
        if is_K_sparse:
            s2array = temp.sum(0) * 2 + K.diagonal()[candidates]

        else:
            s2array = np.sum(temp, axis=0) * 2 + np.diagonal(K)[candidates]

        s2array = s2array/(len(selected) + 1)
        s1array = s1array - s2array

    else:
        if is_K_sparse:
            s1array = s1array - (np.abs(K.diagonal()[candidates]))
        else:
            s1array = s1array - (np.abs(np.diagonal(K)[candidates]))
    return s1array

##############################################################################################################################
# Function choose m of all rows by MMD as per kernelfunc
# ARGS:
# K : kernel matrix
# candidate_indices : array of potential choices for selections, returned values are chosen from these  indices
# t: stopping threshold for finding prototypes
# p: maximum proportion of # of prototypes
# is_K_sparse:  True means K is the pre-computed  csc sparse matrix? False means it is a dense matrix.
# RETURNS: subset of candidate_indices which are selected as prototypes
##############################################################################################################################

def greedy_select_protos(K, candidate_indices, p=0.01, is_K_sparse=False):

    if len(candidate_indices) != np.shape(K)[0]:
        K = K[:,candidate_indices][candidate_indices,:]

    n = len(candidate_indices)
    
    if is_K_sparse:
        colsum = 2*np.array(K.sum(0)).ravel() / n
    else:
        colsum = 2*np.sum(K, axis=0) / n

    selected = np.array([], dtype=int)
    stopping_cond = True
    prev_value = 0

    while True:
        argmax = -1
        candidates = np.setdiff1d(range(n), selected)

        s1array = cost_function(K, selected, colsum, candidates)
        argmax = candidates[np.argmax(s1array)]

        selected = np.append(selected, argmax)
        KK = K[selected, :][:, selected]
        if is_K_sparse:
            KK = KK.todense()

        # inverse_of_prev_selected = np.linalg.inv(KK)  # shortcut

        # print('f_i value of PROTOTYPE is ', np.max(s1array))
        # print('f_i value of DIFF_PROTOTYPE is ', np.max(s1array) - prev_value)
        if len(selected) >= len(K[0])*p:
            stopping_cond = False
        if stopping_cond == True:
            prev_value = np.max(s1array)
            continue
        else: 
            break
    return candidate_indices[selected]

##############################################################################################################################
# function to select criticisms
# ARGS:
# K: Kernel matrix
# selectedprotos: prototypes already selected
# c: stopping threshold for finding criticisms
# p: proportion of number of criticisms
# reg: regularizer type.
# is_K_sparse:  True means K is the pre-computed  csc sparse matrix? False means it is a dense matrix.
# RETURNS: indices selected as criticisms
##############################################################################################################################
def select_criticism_regularized(K, selectedprotos, c=-2, p=0.01, reg='logdet', is_K_sparse=True):

    n = np.shape(K)[0]
    if reg in ['None','logdet','iterative']:
        pass
    else:
        print("wrong regularizer :" + reg)
        exit(1)
    options = dict()

    selected = np.array([], dtype=int)
    candidates2 = np.setdiff1d(range(n), selectedprotos)
    inverse_of_prev_selected = None  # should be a matrix
    # print('proportion of # of criticism is ', p)
    if is_K_sparse:
        colsum = np.array(K.sum(0)).ravel()/n
    else:
        colsum = np.sum(K, axis=0)/n

    stopping_cond_1 = True
    stopping_cond_2 = True
    prev_value = 0

    for i in range(math.ceil(n*p)):
        maxx = -sys.float_info.max
        argmax = -1
        candidates = np.setdiff1d(candidates2, selected)

        s1array = colsum[candidates]

        temp = K[selectedprotos, :][:, candidates]
        if is_K_sparse:
            s2array = temp.sum(0)
        else:
            s2array = np.sum(temp, axis=0)

        s2array = s2array / (len(selectedprotos))
        s1array = np.abs(s1array - s2array)

        log_determinant = lambda x: np.log(np.linalg.det(K[np.append(selected, x), :][:, np.append(selected, x)]))
        log_det_vec = np.vectorize(log_determinant)

        if reg == 'logdet':
            if inverse_of_prev_selected is not None: # first call has been made already
                temp = K[selected, :][:, candidates]
                if is_K_sparse:
                    # temp2 = temp.transpose().dot(inverse_of_prev_selected)
                    # regularizer = temp.transpose().multiply(temp2)
                    # regcolsum = regularizer.sum(1).ravel()# np.sum(regularizer, axis=0)
                    # regularizer = np.abs(K.diagonal()[candidates] - regcolsum)
                    regularizer = log_det_vec(candidates)
                else:
                    # hadamard product
                    # temp2 = np.array(np.dot(inverse_of_prev_selected, temp))
                    # regularizer = temp2 * temp
                    # regcolsum = np.sum(regularizer, axis=0)
                    # regularizer = np.log(np.abs(np.diagonal(K)[candidates] - regcolsum))
                    regularizer = log_det_vec(candidates)
                s1array = s1array + regularizer
            else:
                if is_K_sparse:
                    s1array = s1array - np.log(np.abs(K.diagonal()[candidates]))
                else:
                    s1array = s1array - np.log(np.abs(np.diagonal(K)[candidates]))
        argmax = candidates[np.argmax(s1array)]
        maxx = np.max(s1array)

        selected = np.append(selected, argmax)
        if reg == 'logdet':
            KK = K[selected,:][:,selected]
            if is_K_sparse:
                KK = KK.todense()

            inverse_of_prev_selected = np.linalg.inv(KK) # shortcut
        if reg == 'iterative':
            selectedprotos = np.append(selectedprotos, argmax)

        # print('f_i value of PREV_CRITICISM is ', prev_value)
        # print('the number of f_i is ', len(s1array))
        # print('f_i value of CRITICISM is ', np.max(s1array))
        # print('f_i value of DIFF_CRITICISM is ', np.max(s1array) - prev_value)
        if np.max(s1array) - prev_value < c:
            stopping_cond_1 = False
        if len(selected) >= n*p:
            stopping_cond_2 = False
        if stopping_cond_1 == True and stopping_cond_2 == True:
            prev_value = np.max(s1array)
            continue
        else: 
            break
        
    return selected

def plot(table, protos, critics):
    plt.scatter(table[:,0],table[:,1], marker='s', c='r')
    plt.scatter(table[protos,0],table[protos,1], marker='*', c='b')
    plt.scatter(table[critics,0],table[critics,1], marker='o', c='g')
    plt.grid(True)
    plt.show()
    
def main(filename='data/test2.txt', t=-100, c=-100, p=0.01):
    # data = pd.read_csv(filename)
    data = pd.read_csv(filename, sep='\t', dtype='d')
    sampled_data = data.sample(frac=0.2)

    table = kernel(data)
    protos = greedy_select_protos(table, np.array(range(np.shape(table)[0])), p)
    print('prototypes: ', protos)
    percentage_protos = len(protos)/np.shape(table)[0]
    critics = select_criticism_regularized(table, protos, c=c, p=percentage_protos/4, reg='logdet', is_K_sparse=False)
    print('criticisms: ', critics)
    print('##############################')
    print('total # of prototype is ', len(protos))
    print('total # of criticism is ', len(critics))
    print('##############################')
    plot(data.to_numpy(), protos, critics)

    return protos, critics

if __name__ == "__main__":
    main()
