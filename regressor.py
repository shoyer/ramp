from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor

class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = DummyRegressor()

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
