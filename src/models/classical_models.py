import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)

def evaluate_predictions(y_true, y_pred, y_prob, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    return {
        'model': model_name,
        'accuracy': round(float(acc), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'macro_f1': round(float(macro_f1), 4),
        'weighted_f1': round(float(weighted_f1), 4),
        'roc_auc': round(float(roc_auc), 4),
        'pr_auc': round(float(pr_auc), 4),
        'confusion_matrix': cm
    }

def train_classical_baselines(data_dir="data/processed", results_dir="results"):
    os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
    
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    feature_names = joblib.load(os.path.join(data_dir, "feature_names.joblib"))
    
    scale_pos = (len(y_train) - y_train.sum()) / max(1, y_train.sum())
    
    models = {
        'Logistic Regression (Baseline)': LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42
        ),
        'Random Forest (Balanced)': RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_split=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'XGBoost Classifier': XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.08,
            scale_pos_weight=scale_pos, eval_metric='logloss',
            random_state=42, n_jobs=-1
        )
    }
    
    test_metrics = []
    val_metrics = []
    trained_models = {}
    
    for name, clf in models.items():
        print(f"\n--- Training {name} ---")
        clf.fit(X_train, y_train)
        trained_models[name] = clf
        
        # Validation evaluation
        val_prob = clf.predict_proba(X_val)[:, 1]
        val_pred = (val_prob >= 0.5).astype(int)
        val_res = evaluate_predictions(y_val, val_pred, val_prob, name)
        val_metrics.append(val_res)
        print(f"Val Macro-F1: {val_res['macro_f1']}, ROC-AUC: {val_res['roc_auc']}")
        
        # Test evaluation
        test_prob = clf.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= 0.5).astype(int)
        test_res = evaluate_predictions(y_test, test_pred, test_prob, name)
        test_metrics.append(test_res)
        print(f"Test Accuracy: {test_res['accuracy']}, Macro-F1: {test_res['macro_f1']}, ROC-AUC: {test_res['roc_auc']}")
        
        # Plot confusion matrix
        cm = np.array(test_res['confusion_matrix'])
        plt.figure(figsize=(4.5, 3.8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal (0)', 'Failed (1)'],
                    yticklabels=['Normal (0)', 'Failed (1)'])
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Ground Truth")
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(os.path.join(results_dir, "plots", f"cm_{safe_name}.png"), dpi=200)
        plt.close()
        
    # Save metrics
    with open(os.path.join(results_dir, "metrics", "classical_val_metrics.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)
    with open(os.path.join(results_dir, "metrics", "classical_test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
        
    # Save trained models
    os.makedirs("models_saved", exist_ok=True)
    joblib.dump(trained_models['Random Forest (Balanced)'], "models_saved/random_forest.joblib")
    joblib.dump(trained_models['XGBoost Classifier'], "models_saved/xgboost.joblib")
    joblib.dump(trained_models['Logistic Regression (Baseline)'], "models_saved/logistic_regression.joblib")
    
    print("\nClassical baseline models successfully trained and evaluated.")
    return test_metrics

if __name__ == '__main__':
    train_classical_baselines()
