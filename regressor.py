from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = make_pipeline(
                StandardScaler(),
                PCA(n_components=100),
                LinearRegression()
                # Earth(max_terms=20, max_degree=10)
        )


    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
