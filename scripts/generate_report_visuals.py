import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix

os.makedirs('results/plots', exist_ok=True)

# Set high-DPI academic plotting style with serif fonts
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

print("Loading test predictions...")
df = pd.read_parquet('data/processed/test_predictions.parquet')
y_true = df['y_true'].values

models = [
    ('Logistic Regression (Baseline)', 'logistic_regression_baseline_prob', '#7f7f7f', '--'),
    ('Random Forest (Balanced)', 'random_forest_balanced_prob', '#2ca02c', '-.'),
    ('XGBoost Classifier', 'xgboost_classifier_prob', '#1f77b4', ':'),
    ('Sequential BiLSTM (Trajectory Only)', 'sequence_bilstm_prob', '#9467bd', '-.'),
    ('Proposed Multi-View Fusion Network', 'fusion_prob', '#d62728', '-')
]

# 1. Multi-Model ROC Curves
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
for name, prob_col, color, ls in models:
    y_prob = df[prob_col].values
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    lw = 2.5 if 'Fusion' in name else 1.8
    ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})", color=color, linestyle=ls, linewidth=lw)

ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random Chance (AUC = 0.5000)')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.02])
ax.set_xlabel('False Positive Rate (1 - Specificity)')
ax.set_ylabel('True Positive Rate (Failure Recall / Sensitivity)')
ax.set_title('Receiver Operating Characteristic (ROC) on 1,996 Unseen Test Trials')
ax.legend(loc='lower right', frameon=True, framealpha=0.9)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
fig.savefig('results/plots/roc_curves_all_models.png', dpi=300)
plt.close(fig)
print("Saved results/plots/roc_curves_all_models.png")

# 2. Multi-Model Precision-Recall Curves
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
baseline_prevalence = np.mean(y_true)
for name, prob_col, color, ls in models:
    y_prob = df[prob_col].values
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    lw = 2.5 if 'Fusion' in name else 1.8
    ax.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.4f})", color=color, linestyle=ls, linewidth=lw)

ax.axhline(baseline_prevalence, color='k', linestyle='--', alpha=0.4, label=f'Class Prevalence ({baseline_prevalence:.3f})')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.02])
ax.set_xlabel('Failure Recall (Sensitivity)')
ax.set_ylabel('Precision (Positive Predictive Value)')
ax.set_title('Precision-Recall (PR) Curves on 1,996 Unseen Test Trials')
ax.legend(loc='upper right', frameon=True, framealpha=0.9)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
fig.savefig('results/plots/pr_curves_all_models.png', dpi=300)
plt.close(fig)
print("Saved results/plots/pr_curves_all_models.png")

# 3. Dissected Confusion Matrix Grid (RF, XGBoost, Fusion)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
key_models = [
    ('Random Forest (Balanced)', 'random_forest_balanced_pred', axes[0]),
    ('XGBoost Classifier', 'xgboost_classifier_pred', axes[1]),
    ('Proposed Multi-View Fusion', 'fusion_pred', axes[2])
]

for title, pred_col, ax in key_models:
    cm = confusion_matrix(y_true, df[pred_col].values)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Normal (0)', 'Failure (1)'],
                yticklabels=['Normal (0)', 'Failure (1)'],
                annot_kws={'size': 13, 'weight': 'bold'})
    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    # Calculate recall and accuracy
    rec = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    acc = (cm[0, 0] + cm[1, 1]) / cm.sum()
    ax.text(0.5, -0.22, f"Accuracy: {acc*100:.2f}% | Failure Recall: {rec*100:.2f}%\nFN (Missed Failures): {cm[1,0]}", 
            ha='center', va='center', transform=ax.transAxes, fontsize=10, 
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8fafc', edgecolor='#cbd5e1'))

plt.tight_layout()
fig.savefig('results/plots/confusion_matrix_grid.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("Saved results/plots/confusion_matrix_grid.png")

# 4. Dataset Class Distribution & Partition Breakdown
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

# Class balance
labels = ['Normal / Success (0)', 'Task Failure (1)']
sizes = [9231, 2764]
colors = ['#3b82f6', '#ef4444']
explode = (0, 0.08)
wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                                  autopct='%1.2f%%', startangle=140, pctdistance=0.75,
                                  wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2))
for autotext in autotexts:
    autotext.set_fontsize(11)
    autotext.set_weight('bold')
    autotext.set_color('white')
ax1.set_title('Overall Class Balance (11,995 Trials)\nRatio: 3.34:1 (76.96% vs 23.04%)', fontweight='bold')

# Grouped Split sizes
splits = ['Training Split\n(8,298 trials, 845 tasks)', 
          'Validation Split\n(1,701 trials, 172 tasks)', 
          'Test Split\n(1,996 trials, 869 tasks)']
counts = [8298, 1701, 1996]
bar_colors = ['#0284c7', '#0d9488', '#f59e0b']
bars = ax2.bar(splits, counts, color=bar_colors, edgecolor='#0f172a', width=0.55)
for bar in bars:
    yval = bar.get_height()
    pct = yval / sum(counts) * 100
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 150, f"{yval:,}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_ylim(0, 10000)
ax2.set_ylabel('Number of Execution Trials')
ax2.set_title('Grouped Task Splitting (0% Task Overlap Leakage)', fontweight='bold')
ax2.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
fig.savefig('results/plots/class_distribution.png', dpi=300)
plt.close(fig)
print("Saved results/plots/class_distribution.png")

# 5. Modality Ablation Study Comparison
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
ablation_configs = [
    'Sequence-Only BiLSTM\n(Tool Transitions Only)',
    'Tabular Telemetry Only\n(XGBoost Gradient Boosted)',
    'Tabular Telemetry Only\n(Random Forest Ensemble)',
    'Multi-View Fusion Network\n(Dense Telemetry + Tool Sequence)'
]
recalls = [0.6727 * 100, 0.6705 * 100, 0.5795 * 100, 0.7295 * 100]
praucs = [0.5936 * 100, 0.7132 * 100, 0.7128 * 100, 0.6949 * 100]

x = np.arange(len(ablation_configs))
width = 0.35

rects1 = ax.bar(x - width/2, recalls, width, label='Failure Recall (%)', color='#e11d48', edgecolor='#881337')
rects2 = ax.bar(x + width/2, praucs, width, label='PR-AUC (%)', color='#2563eb', edgecolor='#1e3a8a')

ax.set_ylabel('Metric Score (%)')
ax.set_title('Modality Ablation Analysis: Telemetry vs. Sequence vs. Multi-View Fusion', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(ablation_configs, fontsize=9.5)
ax.set_ylim(0, 90)
ax.legend(loc='upper left', frameon=True)
ax.grid(axis='y', linestyle='--', alpha=0.5)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
fig.savefig('results/plots/ablation_study_chart.png', dpi=300)
plt.close(fig)
print("Saved results/plots/ablation_study_chart.png")
print("All 5 high-resolution research figures generated successfully!")
