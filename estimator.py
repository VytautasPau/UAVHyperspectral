import numpy as np
from sklearn.base import BaseEstimator


class NeuralNetEstimator(BaseEstimator):
    def __init__(self, model=None, device="cpu", summary=False):
        self.model = model
        self.device = device
        self.summary = summary
        if self.summary:
            self.model.summary()

    def fit(self, X, y=None):
        self.model.to(self.device)
        self.model.run(X, y)
        return self

    def predict(self, X):
        self.model.test(X)
        pass

