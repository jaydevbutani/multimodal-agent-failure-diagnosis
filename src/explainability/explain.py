import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

def run_explainability_analysis(data_dir="data/processed", results_dir="results"):
    print("="*60)
    print("RUNNING EXPLAINABILITY & ROOT-CAUSE EVIDENCE ATTRIBUTION")
    print("="*60)
    
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)
    
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    feature_names = joblib.load(os.path.join(data_dir, "feature_names.joblib"))
    
    rf_model = joblib.load("models_saved/random_forest.joblib")
    
    # Feature Importances from Random Forest
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    top_n = 15
    top_features = [feature_names[i] for i in sorted_idx[:top_n]]
    top_importances = importances[sorted_idx[:top_n]]
    
    plt.figure(figsize=(9, 5))
    sns.barplot(x=top_importances, y=top_features, palette="viridis")
    plt.title("Top 15 Most Predictive Telemetry Features for Agent Failure Diagnosis")
    plt.xlabel("Random Forest Gini Feature Importance")
    plt.ylabel("Execution Telemetry / Trajectory Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plots", "feature_importance_top15.png"), dpi=200)
    plt.close()
    
    # Save top features table
    feat_df = pd.DataFrame({
        'Rank': range(1, top_n + 1),
        'Feature': top_features,
        'Importance': [round(float(v), 5) for v in top_importances],
        'Type': ['DERIVED FEATURE' if 'DERIVED' in f else 'Original Telemetry' for f in top_features]
    })
    feat_df.to_csv(os.path.join(results_dir, "metrics", "top_predictive_features.csv"), index=False)
    print("Top 10 Diagnostic Signals:")
    for _, r in feat_df.head(10).iterrows():
        print(f"  {r['Rank']}. {r['Feature']} (Imp: {r['Importance']}, Type: {r['Type']})")
        
    # SHAP TreeExplainer on a sample of test data
    print("\nComputing TreeExplainer SHAP values on test sample...")
    sample_indices = np.random.choice(len(X_test), size=min(300, len(X_test)), replace=False)
    X_sample = X_test[sample_indices]
    
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample)
    
    # SHAP values for class 1 (Failure)
    shap_vals_failure = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values[1]
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_failure, X_sample, feature_names=feature_names, show=False, max_display=12)
    plt.title("SHAP Summary Plot: Impact of Telemetry Signals on Predicted Failure Probability", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plots", "shap_summary_failure.png"), dpi=200)
    plt.close()
    
    print("Explainability analysis successfully generated and plots saved.")

if __name__ == '__main__':
    run_explainability_analysis()
