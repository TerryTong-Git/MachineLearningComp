# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from tqdm_joblib import tqdm_joblib
from tqdm import tqdm
import itertools
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('./'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
train_df = pd.read_csv(
    "./winter-2025-machine-learning-competition-1-p/train.csv")
test_df = pd.read_csv(
    "./winter-2025-machine-learning-competition-1-p/test.csv", index_col=[0])

# %%
len(train_df)

# %%
len(test_df)

# %%

# %%

y = train_df.pop("Label").to_numpy()
X = train_df.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=42)
model = xgb.XGBClassifier(use_label_encoder=False,
                          eval_metric='logloss',
                          random_state=42,
                          verbose=2,
                          tree_method='gpu_hist',
                          n_gpus=-1
                          )

param_grid = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.005, 0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3, 5],
    'reg_alpha': [0, 0.01, 0.1, 1.0],
    'reg_lambda': [1.0, 1.5, 2.0]
}

grid_search = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           scoring='roc_auc',
                           cv=3,
                           n_jobs=8,
                           verbose=2)
param_combinations = list(itertools.product(*param_grid.values()))
total_iter = len(param_combinations) * grid_search.cv
with tqdm_joblib(tqdm(desc="GridSearchCV Progress", total=total_iter)) as progress_bar:
    grid_search.fit(X_train, y_train)


print("Best parameters found: ", grid_search.best_params_)
print("Best CV ROC AUC: ", grid_search.best_score_)

best_model = grid_search.best_estimator_

y_pred_proba = best_model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred_proba)
print("Test ROC AUC: ", auc)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# %%


# %%
# print(train_y.shape, train_X.shape)

# %%
# clf.fit(train_X, train_y)

# %%
# pred_train = clf.predict_proba(train_X)[:, 1] ## prediction on the training dataset
# pred_train

# %%
# roc_auc_score(train_y, pred_train, max_fpr=0.01) # don't forget the 0.01, this will give you the head of the curve


# %%
pred_test = best_model.predict_proba(test_df.to_numpy())[:, 1]
pred_test

# %%
submission = pd.DataFrame({'id': test_df.index, 'Label': pred_test})
submission

# %%
submission.set_index('id').to_csv("submission.csv")

# %%
