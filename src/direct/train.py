import pickle
import sklearn.linear_model


class DirectTrainer:
    def __init__(self):
        self.model = None

    def load_data(self):
        train = pickle.load(open("../../data/train_data.pkl", "rb"))
        test = pickle.load(open("../../data/test_data.pkl", "rb"))
        X_train = train.iloc[:, 1:301]
        Y_train = train.iloc[:, 309:310]
        X_test = test.iloc[:, 1:301]
        Y_test = test.iloc[:, 309:310]
        return X_train, Y_train, X_test, Y_test

    def dump_model(self):
        pickle.dump(self.model, open("../../data/direct_model.pkl", "wb"))

    def generate_model(self):
        X_train, Y_train, X_test, Y_test = self.load_data()
        self.model = sklearn.linear_model.Ridge().fit(X_train, Y_train)
        self.dump_model()


DirectTrainer().generate_model()
