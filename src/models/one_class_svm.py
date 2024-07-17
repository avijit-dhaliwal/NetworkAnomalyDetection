from sklearn.svm import OneClassSVM

class OneClassSVMDetector:
    def __init__(self, config):
        self.config = config
        self.model = OneClassSVM(**config['one_class_svm_params'])

    def train(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)