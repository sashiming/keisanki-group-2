from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor

# 前半パート...文章データから8種類の感情数値を推定

def _import_train_data():
    train_file = Path(__file__).parents[2].joinpath('data', 'train_data.pkl')
    with train_file.open(mode='rb') as f:
        train = pickle.load(f)
    X = train.iloc[:, 1:301]
    y = train.iloc[:, 301:309]
    return X, y

def _dump_model(filename, model):
    model_path = Path(__file__).parents[2].joinpath('models', filename)
    with model_path.open(mode='wb') as f:
        pickle.dump(model, f)

def ridge_model():
    X_train, y_train = _import_train_data()
    params = {'estimator__alpha': [0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0]}
    gscv = GridSearchCV(MultiOutputRegressor(Ridge()),
                        param_grid=params, cv=5, scoring='neg_mean_squared_error')
    gscv.fit(X_train, y_train)
    best_model = gscv.best_estimator_
    print(' ----- ridge -----')
    print('best params:', gscv.best_params_)
    print('best score:', gscv.best_score_)
    _dump_model('emo_ridge_best.pkl', best_model)

def xgboost_model():
    X_train, y_train = _import_train_data()
    params = {'eta': [0.01, 0.1, 0.2],
              'gamma': [0, 0.1],
              'max_depth': [2, 4],}
    gscv = GridSearchCV(XGBRegressor(),
                        param_grid=params, cv=3, verbose=3, scoring='neg_mean_squared_error')
    gscv.fit(X_train, y_train)
    best_model = gscv.best_estimator_
    print('best params:', gscv.best_params_)
    print('best score:', gscv.best_score_)
    _dump_model('emo_xgboost_best.pkl', best_model)

def neuralnet_model():
    X_train, y_train = _import_train_data()
    model = MLPRegressor(hidden_layer_sizes=(100,50,20), max_iter=1000)
    model.fit(X_train, y_train)
    _dump_model('emo_neuralnet.pkl', model)

ridge_model()
xgboost_model()
neuralnet_model()