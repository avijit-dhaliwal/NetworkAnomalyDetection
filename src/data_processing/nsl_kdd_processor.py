import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class NSLKDDProcessor:
    def __init__(self, config):
        self.config = config

    def load_data(self, filepath):
        columns = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
        df = pd.read_csv(filepath, names=columns, header=None)
        return df

    def preprocess(self, df):
        # Handle categorical features
        cat_features = ['protocol_type', 'service', 'flag']
        le = LabelEncoder()
        for feature in cat_features:
            df[feature] = le.fit_transform(df[feature])

        # Normalize numerical features
        num_features = df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        df[num_features] = scaler.fit_transform(df[num_features])

        # Encode labels
        df['label'] = df['label'].map({'normal': 0, 'anomaly': 1})

        return df