import pickle
from sklearn.linear_model import Ridge
from sklearn import svm
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
import numpy as np

import pandas as pd


class DirectTrainer:
    def __init__(self):
        self.model = None

    def load_data(self):
        train = pickle.load(open("../../data/train_data.pkl", "rb"))
        test = pickle.load(open("../../data/validation_data.pkl", "rb"))
        X_train = train.iloc[:, 1:301]
        Y_train = train.iloc[:, 309:310]
        X_test = test.iloc[:, 1:301]
        Y_test = test.iloc[:, 309:310]
        return X_train, Y_train, X_test, Y_test

    def dump_model(self):
        pickle.dump(self.model, open("../../data/direct_model.pkl", "wb"))

    def generate_model(self):
        X_train, Y_train, X_test, Y_test = self.load_data()
        best_score = 0
        for h in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
            model = Ridge(alpha=h, random_state=18).fit(X_train, Y_train)
            score = self.evaluate_model(model, X_test, Y_test)
            # print(h, score)
            if best_score < score:
                best_score = score
                self.model = model
        self.dump_model()

    def evaluate_model(self, model, X_test, Y_test):
        Y_predict = pd.DataFrame(data=model.predict(X_test))
        return metrics.r2_score(Y_test, Y_predict)


DirectTrainer().generate_model()
# r2_score = 0.30488
