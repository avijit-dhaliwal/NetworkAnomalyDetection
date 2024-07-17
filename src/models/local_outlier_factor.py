from sklearn.neighbors import LocalOutlierFactor

class LOFDetector:
    def __init__(self, config):
        self.config = config
        self.model = LocalOutlierFactor(**config['lof_params'])

    def fit_predict(self, X):
        return self.model.fit_predict(X)