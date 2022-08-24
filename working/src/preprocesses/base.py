from sklearn.base import BaseEstimator, TransformerMixin


class BaseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.n_features_in_ = 0

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self
