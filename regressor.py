from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = make_pipeline(
                StandardScaler(),
                KernelPCA(n_components=600, kernel='poly', degree=3),
                # LinearRegression()
                ElasticNet(alpha=3e-4, l1_ratio=0.1),
                # RandomForestRegressor()
                # GradientBoostingRegressor(n_estimators=200, max_features="sqrt", max_depth=5)
                # Earth(max_terms=20, max_degree=10)
        )


    def fit(self, X, y):
        self.clf.fit(X, y.ravel())

    def predict(self, X):
        return self.clf.predict(X)
