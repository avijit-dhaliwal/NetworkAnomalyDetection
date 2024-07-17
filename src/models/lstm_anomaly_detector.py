import tensorflow as tf
from tensorflow.keras import layers, Model

class LSTMAnomalyDetector:
    def __init__(self, config):
        self.config = config
        self.model = self._build_model()

    def _build_model(self):
        input_dim = self.config['input_dim']
        timesteps = self.config['timesteps']
        
        model = tf.keras.Sequential([
            layers.LSTM(64, input_shape=(timesteps, input_dim), return_sequences=True),
            layers.LSTM(32, return_sequences=False),
            layers.Dense(input_dim)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X, epochs=50, batch_size=32):
        self.model.fit(X, X, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    def predict(self, X):
        reconstructed = self.model.predict(X)
        mse = tf.keras.losses.MeanSquaredError()
        reconstruction_error = mse(X, reconstructed).numpy()
        threshold = np.mean(reconstruction_error) + 2 * np.std(reconstruction_error)
        return (reconstruction_error > threshold).astype(int)