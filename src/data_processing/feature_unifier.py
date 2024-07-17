import pandas as pd

def unify_features(nsl_kdd_df, cicids2017_df):
    # Identify common features
    common_features = list(set(nsl_kdd_df.columns) & set(cicids2017_df.columns))
    
    # Add dataset identifier
    nsl_kdd_df['dataset'] = 'NSL-KDD'
    cicids2017_df['dataset'] = 'CICIDS2017'
    
    # Combine datasets using only common features
    unified_df = pd.concat([nsl_kdd_df[common_features + ['dataset']], 
                            cicids2017_df[common_features + ['dataset']]], 
                           axis=0, ignore_index=True)
    
    return unified_df