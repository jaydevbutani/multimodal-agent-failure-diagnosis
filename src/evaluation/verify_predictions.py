import os
import sys
sys.path.insert(0, os.getcwd())
import json
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from src.models.deep_models import SequenceTrajectoryModel, MultimodalAgentFailureModel, encode_sequence

def verify_and_recalculate_metrics(data_dir="data/processed", models_dir="models_saved", results_dir="results"):
    print("="*65)
    print("PHASE 6: AUDIT AND RECALCULATION OF ALL TEST METRICS")
    print("="*65)
    
    # Load test split
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    df_test = pd.read_parquet(os.path.join(data_dir, "test.parquet"))
    vocab = joblib.load(os.path.join(data_dir, "tool_vocab.joblib"))
    
    # Prepare sequence tensor
    seq_encoded = [encode_sequence(s, vocab, max_len=50) for s in df_test['tool_call_sequence']]
    seq_tensor = torch.tensor(seq_encoded, dtype=torch.long)
    tab_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    models = {
        'Logistic Regression (Baseline)': joblib.load(os.path.join(models_dir, "logistic_regression.joblib")),
        'Random Forest (Balanced)': joblib.load(os.path.join(models_dir, "random_forest.joblib")),
        'XGBoost Classifier': joblib.load(os.path.join(models_dir, "xgboost.joblib"))
    }
    
    verified_results = []
    predictions_df = pd.DataFrame({
        'trial_id': df_test['trial_id'],
        'task_id': df_test['task_id'],
        'model': df_test['model'],
        'benchmark': df_test['benchmark'],
        'y_true': y_test
    })
    
    # 1. Classical Models
    for name, clf in models.items():
        probs = clf.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        safe_key = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        predictions_df[f'{safe_key}_prob'] = np.round(probs, 4)
        predictions_df[f'{safe_key}_pred'] = preds
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        macro_f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
        roc_auc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        cm = confusion_matrix(y_test, preds).tolist()
        
        verified_results.append({
            'Model Architecture': name,
            'Test Accuracy': round(float(acc), 4),
            'Precision': round(float(prec), 4),
            'Recall': round(float(rec), 4),
            'Macro-F1': round(float(macro_f1), 4),
            'Weighted-F1': round(float(weighted_f1), 4),
            'ROC-AUC': round(float(roc_auc), 4),
            'PR-AUC': round(float(pr_auc), 4),
            'Confusion Matrix': cm
        })
        
    # 2. Sequence BiLSTM
    model_seq = SequenceTrajectoryModel(vocab_size=len(vocab))
    model_seq.load_state_dict(torch.load(os.path.join(models_dir, "sequence_bilstm.pt")))
    model_seq.eval()
    with torch.no_grad():
        out_seq = model_seq(seq_tensor)
        probs_seq = torch.sigmoid(out_seq).numpy()
    preds_seq = (probs_seq >= 0.5).astype(int)
    predictions_df['sequence_bilstm_prob'] = np.round(probs_seq, 4)
    predictions_df['sequence_bilstm_pred'] = preds_seq
    
    verified_results.append({
        'Model Architecture': 'Sequential BiLSTM (Trajectory Only)',
        'Test Accuracy': round(float(accuracy_score(y_test, preds_seq)), 4),
        'Precision': round(float(precision_score(y_test, preds_seq, zero_division=0)), 4),
        'Recall': round(float(recall_score(y_test, preds_seq, zero_division=0)), 4),
        'Macro-F1': round(float(f1_score(y_test, preds_seq, average='macro', zero_division=0)), 4),
        'Weighted-F1': round(float(f1_score(y_test, preds_seq, average='weighted', zero_division=0)), 4),
        'ROC-AUC': round(float(roc_auc_score(y_test, probs_seq)), 4),
        'PR-AUC': round(float(average_precision_score(y_test, probs_seq)), 4),
        'Confusion Matrix': confusion_matrix(y_test, preds_seq).tolist()
    })
    
    # 3. Proposed Multimodal / Multi-View Fusion Network
    model_fusion = MultimodalAgentFailureModel(tab_dim=X_test.shape[1], vocab_size=len(vocab))
    model_fusion.load_state_dict(torch.load(os.path.join(models_dir, "multimodal_fusion.pt")))
    model_fusion.eval()
    with torch.no_grad():
        out_fusion = model_fusion(tab_tensor, seq_tensor)
        probs_fusion = torch.sigmoid(out_fusion).numpy()
    preds_fusion = (probs_fusion >= 0.5).astype(int)
    predictions_df['fusion_prob'] = np.round(probs_fusion, 4)
    predictions_df['fusion_pred'] = preds_fusion
    
    verified_results.append({
        'Model Architecture': 'Proposed Multi-View Fusion Network (Telemetry + Sequence)',
        'Test Accuracy': round(float(accuracy_score(y_test, preds_fusion)), 4),
        'Precision': round(float(precision_score(y_test, preds_fusion, zero_division=0)), 4),
        'Recall': round(float(recall_score(y_test, preds_fusion, zero_division=0)), 4),
        'Macro-F1': round(float(f1_score(y_test, preds_fusion, average='macro', zero_division=0)), 4),
        'Weighted-F1': round(float(f1_score(y_test, preds_fusion, average='weighted', zero_division=0)), 4),
        'ROC-AUC': round(float(roc_auc_score(y_test, probs_fusion)), 4),
        'PR-AUC': round(float(average_precision_score(y_test, probs_fusion)), 4),
        'Confusion Matrix': confusion_matrix(y_test, preds_fusion).tolist()
    })
    
    # Save predictions file
    predictions_df.to_parquet(os.path.join(data_dir, "test_predictions.parquet"), index=False)
    print(f"Saved exact test predictions for all 5 models to {os.path.join(data_dir, 'test_predictions.parquet')}")
    
    # Output verified comparison table
    verified_df = pd.DataFrame(verified_results)
    table_csv = os.path.join(results_dir, "comparisons", "model_comparison_table.csv")
    verified_df[['Model Architecture', 'Test Accuracy', 'Precision', 'Recall', 'Macro-F1', 'Weighted-F1', 'ROC-AUC', 'PR-AUC']].to_csv(table_csv, index=False)
    
    print("\n--- RECALCULATED & VERIFIED TEST PERFORMANCE TABLE ---")
    print(verified_df[['Model Architecture', 'Test Accuracy', 'Precision', 'Recall', 'Macro-F1', 'ROC-AUC', 'PR-AUC']].to_string(index=False))
    
    return verified_df

if __name__ == '__main__':
    verify_and_recalculate_metrics()
