import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score, classification_report, f1_score

def evaluate_who_and_when_attribution(raw_dir="data/raw/who_when", results_dir="results"):
    print("="*65)
    print("EVALUATING SPECIALIZED ATTRIBUTION BENCHMARK (Who&When)")
    print("="*65)
    
    df_hc = pd.read_parquet(os.path.join(raw_dir, "who_and_when_hc.parquet"))
    df_ag = pd.read_parquet(os.path.join(raw_dir, "who_and_when_ag.parquet"))
    
    # 1. Component Attribution Task on Hand-Crafted (Magentic-One)
    print(f"\n1. Responsible Component Attribution (Hand-Crafted, N={len(df_hc)})")
    # Ground truth targets: WebSurfer, Orchestrator, Assistant, FileSurfer
    # Normalize minor variations
    df_hc['clean_agent'] = df_hc['mistake_agent'].replace({'Websurfer': 'WebSurfer'})
    print("Target distribution:\n", df_hc['clean_agent'].value_counts())
    
    # Extract features from multi-turn history:
    # - total turns
    # - turn count per agent role
    # - code execution presence
    # - mistake step normalized position
    feat_rows = []
    for idx, row in df_hc.iterrows():
        hist = row['history']
        roles = [t.get('name', t.get('role', 'unknown')) for t in hist]
        contents = [str(t.get('content', '')) for t in hist]
        
        role_counts = pd.Series(roles).value_counts()
        feat_rows.append({
            'total_turns': len(hist),
            'orchestrator_turns': role_counts.get('Orchestrator (thought)', 0) + role_counts.get('Orchestrator (-> WebSurfer)', 0),
            'websurfer_turns': role_counts.get('WebSurfer', 0),
            'assistant_turns': role_counts.get('Assistant', 0),
            'filesurfer_turns': role_counts.get('FileSurfer', 0),
            'has_code_turns': sum(1 for c in contents if '```' in c),
            'avg_turn_length': np.mean([len(c) for c in contents]) if contents else 0,
            'max_turn_length': np.max([len(c) for c in contents]) if contents else 0
        })
        
    X_comp = pd.DataFrame(feat_rows)
    y_comp = df_hc['clean_agent'].values
    
    # Cross-validated evaluation (3-fold CV to match minimum class frequency of FileSurfer N=3)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    accs, top2_accs, macro_f1s = [], [], []
    
    classes = np.unique(y_comp)
    for train_idx, test_idx in skf.split(X_comp, y_comp):
        X_tr, X_te = X_comp.iloc[train_idx], X_comp.iloc[test_idx]
        y_tr, y_te = y_comp[train_idx], y_comp[test_idx]
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight='balanced')
        clf.fit(X_tr, y_tr)
        
        preds = clf.predict(X_te)
        probs = clf.predict_proba(X_te)
        
        accs.append(accuracy_score(y_te, preds))
        macro_f1s.append(f1_score(y_te, preds, average='macro', zero_division=0))
        
        # Top-2 accuracy
        if len(classes) >= 2:
            # Map probabilities to classes
            top2 = [1 if y_te[i] in clf.classes_[np.argsort(probs[i])[-2:]] else 0 for i in range(len(y_te))]
            top2_accs.append(np.mean(top2))
            
    comp_results = {
        'task': 'Responsible Component Attribution (Task D)',
        'benchmark': 'Who&When (Magentic-One)',
        'samples': len(df_hc),
        'classes': classes.tolist(),
        'top1_accuracy': round(float(np.mean(accs)), 4),
        'top2_accuracy': round(float(np.mean(top2_accs)), 4),
        'macro_f1': round(float(np.mean(macro_f1s)), 4)
    }
    print("Component Attribution Results:", comp_results)
    
    # 2. Step-Level Temporal Fault Localization
    print(f"\n2. Step-Level Temporal Fault Localization (4,092 steps)")
    # For each step in trajectory, predict whether it is the decisive failure point
    step_records = []
    for idx, row in df_hc.iterrows():
        hist = row['history']
        m_step = int(row['mistake_step'])
        for s_idx, turn in enumerate(hist):
            content = str(turn.get('content', ''))
            is_critical = int(s_idx == m_step)
            step_records.append({
                'step_idx': s_idx,
                'normalized_step': s_idx / max(1, len(hist) - 1),
                'content_length': len(content),
                'has_code': int('```' in content),
                'has_error': int(any(w in content.lower() for w in ['error', 'failed', 'traceback', 'exception'])),
                'is_critical_step': is_critical,
                'task_id': row['question_ID']
            })
            
    step_df = pd.DataFrame(step_records)
    print(f"Total steps analyzed: {len(step_df)}, Critical failure points: {step_df['is_critical_step'].sum()}")
    
    X_step = step_df[['step_idx', 'normalized_step', 'content_length', 'has_code', 'has_error']]
    y_step = step_df['is_critical_step'].values
    
    # Evaluate step localization
    clf_step = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42)
    # Grouped split on task_id
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(step_df, groups=step_df['task_id']))
    
    clf_step.fit(X_step.iloc[tr_idx], y_step[tr_idx])
    step_probs = clf_step.predict_proba(X_step.iloc[te_idx])[:, 1]
    
    # Check if predicted top step per task matches true mistake_step
    test_tasks = step_df.iloc[te_idx]['task_id'].unique()
    top1_hits, top3_hits, step_errors = [], [], []
    for t in test_tasks:
        sub = step_df.iloc[te_idx][step_df.iloc[te_idx]['task_id'] == t]
        sub_probs = step_probs[step_df.iloc[te_idx]['task_id'] == t]
        true_step = sub[sub['is_critical_step'] == 1]['step_idx'].values
        if len(true_step) > 0:
            true_s = true_step[0]
            ranked_steps = sub.iloc[np.argsort(-sub_probs)]['step_idx'].values
            top1_hits.append(int(ranked_steps[0] == true_s))
            top3_hits.append(int(true_s in ranked_steps[:3]))
            step_errors.append(abs(ranked_steps[0] - true_s))
            
    temp_results = {
        'task': 'Step-Level Temporal Fault Localization (Task C)',
        'benchmark': 'Who&When',
        'test_tasks_evaluated': len(test_tasks),
        'top1_exact_step_hit_rate': round(float(np.mean(top1_hits)), 4),
        'top3_step_hit_rate': round(float(np.mean(top3_hits)), 4),
        'mean_absolute_step_error': round(float(np.mean(step_errors)), 2)
    }
    print("Temporal Fault Localization Results:", temp_results)
    
    # Save attribution metrics
    os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)
    with open(os.path.join(results_dir, "metrics", "attribution_metrics.json"), "w") as f:
        json.dump({'component_attribution': comp_results, 'temporal_localization': temp_results}, f, indent=2)
        
    print("\nAttribution benchmark evaluation complete.")

if __name__ == '__main__':
    evaluate_who_and_when_attribution()
