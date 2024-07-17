def analyze_attack_patterns(nsl_kdd_df, cicids2017_df):
    nsl_kdd_attacks = nsl_kdd_df[nsl_kdd_df['label'] == 1]['attack_type'].value_counts()
    cicids2017_attacks = cicids2017_df[cicids2017_df['Label'] == 1]['Label'].value_counts()

    common_attacks = set(nsl_kdd_attacks.index) & set(cicids2017_attacks.index)
    new_attacks = set(cicids2017_attacks.index) - set(nsl_kdd_attacks.index)

    return {
        'common_attacks': common_attacks,
        'new_attacks': new_attacks,
        'nsl_kdd_distribution': nsl_kdd_attacks,
        'cicids2017_distribution': cicids2017_attacks
    }

def compare_feature_distributions(nsl_kdd_df, cicids2017_df):
    common_features = list(set(nsl_kdd_df.columns) & set(cicids2017_df.columns))
    distribution_changes = {}

    for feature in common_features:
        nsl_kdd_dist = nsl_kdd_df[feature].describe()
        cicids2017_dist = cicids2017_df[feature].describe()
        
        distribution_changes[feature] = {
            'mean_change': cicids2017_dist['mean'] - nsl_kdd_dist['mean'],
            'std_change': cicids2017_dist['std'] - nsl_kdd_dist['std'],
            'min_change': cicids2017_dist['min'] - nsl_kdd_dist['min'],
            'max_change': cicids2017_dist['max'] - nsl_kdd_dist['max']
        }

    return pd.DataFrame(distribution_changes).T