import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class CICIDS2017Processor:
    def __init__(self, config):
        self.config = config

    def load_data(self, filepath):
        df = pd.read_csv(filepath)
        return df

    def preprocess(self, df):
        # Drop unnecessary columns
        df = df.drop(['Timestamp'], axis=1)

        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        # Normalize numerical features
        num_features = df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        df[num_features] = scaler.fit_transform(df[num_features])

        # Encode labels
        df['Label'] = df['Label'].map(lambda x: 0 if x == 'BENIGN' else 1)

        return df