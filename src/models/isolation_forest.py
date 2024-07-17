from sklearn.ensemble import IsolationForest

class IsolationForestDetector:
    def __init__(self, config):
        self.config = config
        self.model = IsolationForest(**config['isolation_forest_params'])

    def train(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)