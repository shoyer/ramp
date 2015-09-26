from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsRegressor

class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = KNeighborsRegressor()

    def fit(self, X, y):
        self.clf.fit(X[:, 1:], X[:, 0])

    def predict(self, X):
        return self.clf.predict(X[:, 1:])
