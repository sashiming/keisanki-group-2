from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import RidgeCV

def _import_train_data():
    train_file = Path(__file__).parents[2].joinpath('data', 'train_data.pkl')
    with train_file.open(mode='rb') as f:
        train = pickle.load(f)
    return train

def ridge_model():
    train_data = _import_train_data()

print(_import_train_data().head())