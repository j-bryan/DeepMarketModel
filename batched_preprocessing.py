import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class BatchStandardScaler(BaseEstimator, TransformerMixin):
    """
    Implements a standard scaler that works with 3D arrays. All operations are performed along the
    last axis.
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        self.mean_ = X.mean(axis=(0, 1))
        self.std_ = X.std(axis=(0, 1))
        return self

    def transform(self, X, y=None):
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X, y=None):
        return X * self.std_ + self.mean_

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class BatchMinMaxScaler(BaseEstimator, TransformerMixin):
    """
    Implements a min-max scaler that works with 3D arrays. All operations are performed along the
    last axis.
    """
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X, y=None):
        self.min_ = X.min(axis=(0, 1))
        self.max_ = X.max(axis=(0, 1))
        return self

    def transform(self, X, y=None):
        return (X - self.min_) / (self.max_ - self.min_)

    def inverse_transform(self, X, y=None):
        return X * (self.max_ - self.min_) + self.min_

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class BatchRobustScaler(BaseEstimator, TransformerMixin):
    """
    Implements a robust scaler that works with 3D arrays. All operations are performed along the
    last axis.
    """
    def __init__(self):
        self.quantiles_ = None

    def fit(self, X, y=None):
        self.quantiles_ = np.percentile(X, [25, 50, 75], axis=(0, 1))
        return self

    def transform(self, X, y=None):
        q25, q50, q75 = self.quantiles_
        return (X - q50) / (q75 - q25)

    def inverse_transform(self, X, y=None):
        q25, q50, q75 = self.quantiles_
        return X * (q75 - q25) + q50

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class BatchColumnTransformer(BaseEstimator, TransformerMixin):
    """
    Implements a column transformer that works with 3D arrays. All operations are performed along the
    last axis.
    """
    def __init__(self, transformers, remainder="passthrough"):
        self.transformers = transformers
        if remainder != "passthrough":
            raise ValueError("Only 'passthrough' is supported for remainder")

    def fit(self, X, y=None):
        for name, transformer, columns in self.transformers:
            transformer.fit(X[..., columns].reshape(-1, len(columns)))
        return self

    def transform(self, X, y=None):
        batch_size = X.shape[0]
        Xt = X.copy()
        for name, transformer, columns in self.transformers:
            Xt[..., columns] = transformer.transform(X[..., columns].reshape(-1, len(columns))).reshape(batch_size, -1, len(columns))
        return Xt

    def inverse_transform(self, X, y=None):
        batch_size = X.shape[0]
        Xt = X.copy()
        for name, transformer, columns in self.transformers:
            Xt[..., columns] = transformer.inverse_transform(X[..., columns].reshape(-1, len(columns))).reshape(batch_size, -1, len(columns))
        return Xt

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
