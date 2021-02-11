import numpy as np
from numpy.random import sample
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import make_moons, make_circles
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error, r2_score
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from table.proto_critic import kernel, greedy_select_protos, select_criticism_regularized
from table.run_glocal_MART import run_glocal_MART, kernel_function
import lime.lime_tabular
import shap
from math import sqrt
import xgboost as xgb

np.random.seed(1)

#### Dataset ###
data_name = 'wisc_bc_data'
target = 'diagnosis'
data = pd.read_csv('./table/' + data_name + '.csv')
data[target] = data[target].map({'M': 1, 'B': 0})

# data_name = 'biodeg'
# target = 'label'
# data = pd.read_csv('./table/' + data_name + '.csv')

# data_name = 'musk'
# target = 'class'
# data = pd.read_csv('./table/' + data_name + '.csv')

# data_name = 'forest_fires_one_hot'
# target = 'area'
# data = pd.read_csv('./table/' + data_name + '.csv')

# data_name = 'benz'
# target = 'y'
# data = pd.read_csv('./table/' + data_name + '.csv')

#### Data Split ####
X_0 = data.drop([target], axis=1)
Y = data[[target]]
X_train, X_test, y_train, y_test = train_test_split(X_0, Y, test_size=0.3, random_state=42)
X_test_idx = len(X_train)
X = np.concatenate((X_train, X_test), axis=0)
sample_input = X.copy()

### Train Model ###
# for wisc_bc_data / biodeg / musk
rf = RandomForestClassifier().fit(X_train, y_train)
orig_score = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

# for forest_fires_one_hot
# rf = RandomForestRegressor(n_estimators=500, max_depth=10, max_features=5, min_samples_split=10).fit(X_train, y_train)
# orig_score = r2_score(y_test.to_numpy().flatten(), rf.predict(X_test))

# for benz
# rf = RandomForestRegressor(n_estimators=100, max_depth=20, max_features=20, min_samples_split=20).fit(X_train, y_train)
# orig_score = r2_score(y_test.to_numpy().flatten(), rf.predict(X_test))

print('Test score is {}.'.format(orig_score))
preds = rf.predict(sample_input)

#### Calculate GLocal_MART ####
# table = kernel(X)
# protos = greedy_select_protos(table, np.array(range(np.shape(table)[0])), p=0.1)
# percentage_protos = len(protos)/np.shape(table)[0]
# critics = select_criticism_regularized(table, protos, c=-2, p=percentage_protos/4, reg='logdet', is_K_sparse=False)
# proto_critics_idx = np.append(protos, critics)
df_X = pd.DataFrame(X)
#
# glm = run_glocal_MART(input=X_test.to_numpy(), X=df_X, model_function=rf, proto_critics_idx=proto_critics_idx,
#                       kernel_function=kernel_function, classification=True, steps=20)
# np.save('./table/'+ data_name + '_mart', glm)


#### Calculate LIME & SHAP ####
# lime_explainer = lime.lime_tabular.LimeTabularExplainer(X, mode='classification', feature_names=list(X_0.columns),
#                                                         discretize_continuous=False, sample_around_instance = True)
# shap_explainer = shap.TreeExplainer(rf)
# shap_values = shap_explainer.shap_values(sample_input)
# lime = np.array([])
# shap = np.array([])
# for i in range(X_test_idx, sample_input.shape[0]):
#     target = sample_input[i,:]
#     exp = lime_explainer.explain_instance(target, rf.predict_proba, top_labels = 1, num_features=len(list(X_0.columns)))
#     temp = np.array(exp.as_list(exp.available_labels()[0]))
#     # exp = lime_explainer.explain_instance(target, rf.predict, num_features=len(list(X_0.columns)))
#     # temp = np.array(exp.as_list())
#     for col in list(X_0.columns):
#         for idx in range(temp.shape[0]):
#             if col == temp[idx, 0]:
#                 lime = np.append(lime, np.array(temp[idx, 1]))
#     # temp.sort(key = lambda tup:tup[0])
#     # lime = np.append(lime, np.array(temp)[:,1])
#     # shap = np.append(shap, shap_values[preds[i]][i])
#
# lime = lime.reshape((X_test.shape[0], X_test.shape[1]))
# # shap = shap.reshape((X_test.shape[0], X_test.shape[1]))
# shap = shap_values[1][X_test_idx:, :]
# # shap = shap_values[X_test_idx:, :]
#
# np.save('./table/'+ data_name + '_lime', lime)
# np.save('./table/'+ data_name + '_shap', shap)


#### Load Attribution Result ####
glm = np.load('./table/'+ data_name + '_mart.npy')
shap = np.load('./table/' + data_name + '_shap.npy')
lime = np.load('./table/' + data_name + '_lime.npy')

feat_imp = glm

from random import uniform, random, seed, randint
seed(0)
seed_list = [randint(0, 1e10) for i in range(20)]

percentage_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for percentage in percentage_list:
    num = int(X.shape[1] * (1-percentage) + 1)
    diff_result = np.array([])
    for seed_it in seed_list:
        seed(seed_it)

        #### Except benz ####
        random_num = uniform(-1e10, 1e10)
        #### For benz ####
        # random_num = uniform(0, 1)

        for row_idx in range(X_test_idx, len(df_X)):
            max_idx_arr = np.argsort(feat_imp[row_idx - X_test_idx, :])[:num]
            for max_idx in max_idx_arr:
                X[row_idx, max_idx] = random_num
        perm_test_X = X[X_test_idx:, :]

        #### For Classification ####
        score = roc_auc_score(y_test, rf.predict_proba(perm_test_X)[:, 1])
        #### For Regression ####
        # score = r2_score(y_test.to_numpy().flatten(), rf.predict(perm_test_X))

        diff_result = np.append(diff_result, np.array([max(orig_score - score, 0)]))
        # diff_result = np.append(diff_result, np.array([abs(orig_score - score)]))
        X = sample_input.copy()
    print(str(int(percentage * 100)) + "%: " + str(np.average(diff_result)))

