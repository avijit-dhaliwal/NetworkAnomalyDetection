import tensorflow as tf
from tensorflow.keras import layers, Model

class AutoencoderDetector:
    def __init__(self, config):
        self.config = config
        self.model = self._build_model()

    def _build_model(self):
        input_dim = self.config['input_dim']
        encoding_dim = self.config['encoding_dim']

        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
        decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')

        return autoencoder

    def train(self, X, epochs=50, batch_size=32):
        self.model.fit(X, X, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    def predict(self, X):
        reconstructed = self.model.predict(X)
        mse = tf.keras.losses.MeanSquaredError()
        reconstruction_error = mse(X, reconstructed).numpy()
        threshold = np.mean(reconstruction_error) + 2 * np.std(reconstruction_error)
        return (reconstruction_error > threshold).astype(int)