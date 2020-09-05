import numpy as np
from numpy.random import sample
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons, make_circles
from sklearn.metrics import roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from table.proto_critic import kernel, greedy_select_protos, select_criticism_regularized
from table.run_glocal_MART import run_glocal_MART, kernel_function
import lime.lime_tabular
import shap

np.random.seed(1)

### Generate toy dataset ###
n_samples = 1000
X, Y = make_moons(noise=0.12, random_state=0, n_samples=n_samples)
# X, Y = make_circles(noise=0.15, random_state=0, n_samples=n_samples, factor = 0.7)
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42)

#sample_input = np.array([[0.455, 0.893], [0.255, 0.893],[0.7, 0.893],[1, 0.893],[0, 0.893],])
sample_input = X

### Train Model ###
rf = RandomForestClassifier().fit(X_train, y_train)
print('Test score is {}.'
      .format(roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])))
preds = rf.predict(sample_input)

# Calculate GLocal_MART
table = kernel(X)
protos = greedy_select_protos(table, np.array(range(np.shape(table)[0])), p=0.1)
percentage_protos = len(protos)/np.shape(table)[0]
critics = select_criticism_regularized(table, protos, c=-2, p=percentage_protos/4, reg='logdet', is_K_sparse=False)
proto_critics_idx = np.append(protos, critics)
df_X = pd.DataFrame(X)
glm = run_glocal_MART(input=sample_input, X=df_X, model_function=rf, proto_critics_idx=proto_critics_idx, kernel_function=kernel_function, steps = 100)


# ### Calculate LIME & SHAP ###
# lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=['X','Y'], class_names=['Class 0', 'Class 1'], discretize_continuous=False, sample_around_instance = True)
# shap_explainer = shap.TreeExplainer(rf)
# shap_values = shap_explainer.shap_values(sample_input)
# lime_x = np.array([])
# lime_y = np.array([])
# shap_x = np.array([])
# shap_y = np.array([])
# for i in range(sample_input.shape[0]):
#   target = sample_input[i,:]
#   exp = lime_explainer.explain_instance(target, rf.predict_proba, top_labels = 1)
#   temp = exp.as_list(exp.available_labels()[0])
#   temp.sort(key = lambda tup:tup[0])
#   lime_x = np.append(lime_x, temp[0][1])
#   lime_y = np.append(lime_y, temp[1][1])
#   shap_x = np.append(shap_x, shap_values[preds[i]][i][0])
#   shap_y = np.append(shap_y, shap_values[preds[i]][i][1])


### Result ###
result = pd.DataFrame({'x':sample_input[:,0], 'y':sample_input[:,1], 'preds': preds, 'glm_x': glm[:,0], 'glm_y':glm[:,1]})
print('Result')
print(result)

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(18,5))

ax1.scatter(X[Y==0, 0], X[Y==0, 1], s=10, c='#FFB1A0', label='Class 0')
ax1.scatter(X[Y==1, 0], X[Y==1, 1], s=10, c='#A0D5FF', label='Class 1')

sns.despine(); ax1.legend()

cm = matplotlib.cm.get_cmap('RdYlBu')

X1_min, X1_max = X[:, 0].min(), X[:, 0].max()
X2_min, X2_max = X[:, 1].min(), X[:, 1].max()
grid_step = 1000
X1_range, X2_range = np.meshgrid(np.linspace(X1_min, X1_max, grid_step), np.linspace(X2_min, X2_max, grid_step))
X1_X2 = np.c_[X1_range.ravel(), X2_range.ravel()]

pred_prob = rf.predict_proba(X1_X2)[:, 0]
Y = pred_prob.reshape(X1_range.shape)
sc = plt.contourf(X1_range, X2_range, 1-Y, 100, cmap=cm)
ax1.set(xlabel='X', ylabel='Y')
ax2.set(xlabel='X', ylabel='Y')
ax3.set(xlabel='X', ylabel='Y')
ax1.title.set_text("Original data")
ax2.title.set_text("MART")
ax3.title.set_text("Probability of class 0")
sc2 = ax2.scatter(sample_input[:, 0], sample_input[:, 1], c=np.array(result['glm_x']), s=10, vmin=0.0, vmax=1.0, cmap=cm, label = 'Targets')
plt.colorbar(sc)
#plt.colorbar(sc2)
plt.show()