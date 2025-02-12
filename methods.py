
from joblib import Memory
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import KFold
from pathlib import Path
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, HalvingGridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import xgboost as xgb
import itertools
from tqdm.autonotebook import tqdm
from sklearn import svm


def xgboost(X_train, y_train):
    kf = KFold(n_splits=5, shuffle=False)

    model = xgb.XGBClassifier(eval_metric='logloss',
                              random_state=42,
                              tree_method='hist',
                              device="cuda:0"
                              )

    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
    }

    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               scoring='roc_auc',
                               cv=kf,
                               n_jobs=-1,
                               verbose=2)
    grid_search.fit(X_train, y_train)

    print("Best parameters found: ", grid_search.best_params_)
    print("Best CV ROC AUC: ", grid_search.best_score_)

    best_model = grid_search.best_estimator_
    return best_model


def SupportVec(X_train, y_train):
    kf = KFold(n_splits=5, shuffle=False)

    # fit the model and get the separating hyperplane
    clf = svm.SVC(probability=True)
    parameters = {'kernel': ('linear', 'rbf'), 'C': [
        0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1]}

    grid_search = GridSearchCV(estimator=clf,
                               param_grid=parameters,
                               scoring='roc_auc',
                               cv=kf,
                               n_jobs=2,
                               verbose=3)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    return best_model


# Build the pipeline


def pipeline(X_train, y_train):
    kf = KFold(n_splits=5, shuffle=False)
    Path("./cache_dir").mkdir()
    memory = Memory(location='./cachedir', verbose=0)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(score_func=mutual_info_classif, k=20)),
        ('kernel_approx', RBFSampler(gamma=0.1, n_components=300, random_state=42)),
        ('calibrated_classifier', CalibratedClassifierCV(
            base_estimator=SGDClassifier(
                max_iter=5000, tol=1e-4, random_state=42, early_stopping=True, warm_start=True),
            cv=kf))
    ], memory=memory)

    # Define a parameter grid to search over:
    # - The number of features to select from the 30 available.
    # - The gamma value and number of random Fourier components for the RBF approximation.
    # - The number of iterations and tolerance for SGDClassifier.
    param_grid = {
        'feature_selection__k': [15, 20, 25],
        'kernel_approx__gamma': [0.01, 0.1, 0.2],
        'kernel_approx__n_components': [100, 300, 500],
        'calibrated_classifier__base_estimator__max_iter': [5000, 10000],
    }

    # Wrap the pipeline in a grid search with 3-fold cross-validation.
    grid_search = HalvingGridSearchCV(pipeline,
                                      param_grid=param_grid,
                                      cv=kf,
                                      scoring='roc_auc',
                                      verbose=2,
                                      n_jobs=-1)

    # Fit the grid search to the training data.
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model


# # fit the model and get the separating hyperplane using weighted classes
# wclf = svm.SVC(kernel="linear", class_weight={1: 10})
# wclf.fit(X, y)
