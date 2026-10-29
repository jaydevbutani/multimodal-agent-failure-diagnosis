import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_comparative_report(results_dir="results"):
    print("="*60)
    print("COMPILING COMPREHENSIVE COMPARATIVE ANALYSIS")
    print("="*60)
    
    with open(os.path.join(results_dir, "metrics", "classical_test_metrics.json"), "r") as f:
        classical_metrics = json.load(f)
        
    with open(os.path.join(results_dir, "metrics", "deep_test_metrics.json"), "r") as f:
        deep_metrics = json.load(f)
        
    all_models = classical_metrics + deep_metrics
    
    table_rows = []
    for m in all_models:
        table_rows.append({
            'Model Architecture': m['model'],
            'Test Accuracy': m['accuracy'],
            'Precision': m['precision'],
            'Recall': m['recall'],
            'Macro-F1': m['macro_f1'],
            'Weighted-F1': m['weighted_f1'],
            'ROC-AUC': m['roc_auc'],
            'PR-AUC': m['pr_auc']
        })
        
    df_comp = pd.DataFrame(table_rows)
    os.makedirs(os.path.join(results_dir, "comparisons"), exist_ok=True)
    df_comp.to_csv(os.path.join(results_dir, "comparisons", "model_comparison_table.csv"), index=False)
    
    print("\n--- Model Benchmark Performance Comparison (1,996 Test Samples) ---")
    print(df_comp.to_string(index=False))
    
    # Generate Comparison Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_comp))
    width = 0.22
    
    ax.bar(x - width, df_comp['Test Accuracy'], width, label='Accuracy', color='#3498db')
    ax.bar(x, df_comp['Macro-F1'], width, label='Macro-F1', color='#e74c3c')
    ax.bar(x + width, df_comp['ROC-AUC'], width, label='ROC-AUC', color='#2ecc71')
    
    ax.set_ylabel('Score')
    ax.set_title('Comparative Evaluation Across Model Hierarchy on Agent Failure Diagnosis')
    ax.set_xticks(x)
    ax.set_xticklabels(df_comp['Model Architecture'], rotation=15, ha='right', fontsize=9)
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plots", "model_comparison_bar.png"), dpi=200)
    plt.close()
    
    print(f"\nComparative table and visualization successfully saved to {results_dir}/comparisons/ and {results_dir}/plots/")
    return df_comp

if __name__ == '__main__':
    generate_comparative_report()
