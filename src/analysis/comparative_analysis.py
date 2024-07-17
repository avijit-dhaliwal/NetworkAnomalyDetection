import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

def compare_model_performance(models, X, y):
    results = []
    for model_name, model in models.items():
        y_pred = model.predict(X)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
        auc_roc = roc_auc_score(y, y_pred)
        results.append({
            'model': model_name,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc
        })
    return pd.DataFrame(results)

def compare_feature_importance(models, feature_names):
    importance_df = pd.DataFrame(index=feature_names)
    for model_name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importance_df[model_name] = model.feature_importances_
    return importance_df