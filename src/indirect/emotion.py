from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

def _import_train_data():
    train_file = Path(__file__).parents[2].joinpath('data', 'new_train_data.pkl')
    with train_file.open(mode='rb') as f:
        train = pickle.load(f)
    print(train.shape)
    print(train)
    X = train.iloc[:, 1:301]
    y = train.iloc[:, 301:309]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    return X_train, X_test, y_train, y_test

def ridge_model():
    X_train, X_test, y_train, y_test = _import_train_data()
    model = MultiOutputRegressor(Ridge(random_state=42))
    model.fit(X_train, y_train)
    print(model.predict(X_test)[:20])

# _import_train_data()
ridge_model()