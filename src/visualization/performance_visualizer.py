import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_performance(performance_df):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model', y='f1_score', data=performance_df)
    plt.title('Model Performance Comparison')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(importance_df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(importance_df, annot=True, cmap='YlOrRd')
    plt.title('Feature Importance Comparison')
    plt.tight_layout()
    plt.show()