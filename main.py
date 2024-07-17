# main.py
import yaml
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import numpy as np

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_nsl_kdd(path):
    # Define column names for NSL-KDD dataset
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
        'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
        'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
        'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class'
    ]
    
    # Read the file, skipping lines that start with @
    df = pd.read_csv(path, names=columns, comment='@', header=None)
    
    # Convert 'class' column to binary (0 for normal, 1 for attack)
    df['class'] = (df['class'] != 'normal').astype(int)
    
    return df

def load_cicids2017(directory, files):
    dfs = []
    for file in files:
        df = pd.read_csv(os.path.join(directory, file))
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        # Look for 'Label' column (case-insensitive)
        label_col = df.columns[df.columns.str.lower() == 'label']
        if len(label_col) > 0:
            # Create a binary 'class' column (0 for normal, 1 for attack)
            df['class'] = (df[label_col[0]] != 'BENIGN').astype(int)
            dfs.append(df)
        else:
            print(f"Warning: 'Label' column not found in {file}. Skipping this file.")
    
    if not dfs:
        raise ValueError("No valid CICIDS2017 files found.")
    
    return pd.concat(dfs, ignore_index=True)

def preprocess_data(df):
    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Handle missing values
    df = df.dropna()
    
    # Convert categorical variables to numerical
    for col in df.select_dtypes(['object']):
        df[col] = pd.Categorical(df[col]).codes
    
    # Ensure all columns are numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                print(f"Couldn't convert column {col} to numeric. Dropping this column.")
                df = df.drop(col, axis=1)
    
    # Drop any remaining non-numeric columns
    df = df.select_dtypes(include=[np.number])
    
    return df

def main():
    # Load configurations
    data_config = load_config('configs/data_config.yaml')
    model_config = load_config('configs/model_config.yaml')

    # Load NSL-KDD dataset
    nsl_kdd = load_nsl_kdd(data_config['nsl_kdd']['path'])
    print("NSL-KDD dataset loaded. Shape:", nsl_kdd.shape)
    print("NSL-KDD columns:", nsl_kdd.columns.tolist())

    # Load CICIDS2017 dataset
    cicids2017 = load_cicids2017(data_config['cicids2017']['path'], data_config['cicids2017']['files'])
    print("CICIDS2017 dataset loaded. Shape:", cicids2017.shape)
    print("CICIDS2017 columns:", cicids2017.columns.tolist())

    # Preprocess datasets
    nsl_kdd = preprocess_data(nsl_kdd)
    cicids2017 = preprocess_data(cicids2017)

    print("NSL-KDD shape after preprocessing:", nsl_kdd.shape)
    print("CICIDS2017 shape after preprocessing:", cicids2017.shape)

    # Prepare NSL-KDD dataset
    X_nsl = nsl_kdd.drop('class', axis=1)
    y_nsl = nsl_kdd['class']

    # Prepare CICIDS2017 dataset
    X_cicids = cicids2017.drop('class', axis=1)
    y_cicids = cicids2017['class']

    # Combine datasets
    X_combined = pd.concat([X_nsl, X_cicids], axis=0, ignore_index=True)
    y_combined = pd.concat([y_nsl, y_cicids], axis=0, ignore_index=True)

    print("Combined dataset shape:", X_combined.shape)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

    # Train Isolation Forest model
    clf = IsolationForest(**model_config['isolation_forest_params'])
    clf.fit(X_train)

    # Predict anomalies
    y_pred = clf.predict(X_test)
    y_pred = [1 if x == -1 else 0 for x in y_pred]  # Convert to binary (1 for anomaly, 0 for normal)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()