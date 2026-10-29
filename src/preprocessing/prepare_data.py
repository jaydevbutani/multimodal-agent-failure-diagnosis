import os
import math
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def compute_tool_entropy(seq):
    if seq is None or len(seq) == 0:
        return 0.0
    counts = Counter(seq)
    total = len(seq)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log(p)
    return float(entropy)

def engineer_features(df):
    """
    Computes genuine DERIVED FEATURES strictly from individual trial telemetry.
    No cross-row aggregation or target information is used, guaranteeing zero leakage.
    """
    df = df.copy()
    
    # Target definition: Binary failure indicator
    df['is_failed'] = (~df['passed']).astype(int)
    
    # 1. Error rate normalized by total tool calls
    df['DERIVED_error_rate'] = df['error_count'] / np.maximum(1, df['tool_calls_total'])
    
    # 2. Retry ratio normalized by error count
    df['DERIVED_retry_ratio'] = df['retry_count'] / np.maximum(1, df['error_count'])
    
    # 3. Ratio of search calls to edit calls
    df['DERIVED_search_to_edit_ratio'] = df['search_tool_calls'] / np.maximum(1, df['edit_tool_calls'])
    
    # 4. Edit churn: patch lines modified per edited file
    df['DERIVED_edit_churn'] = df['patch_size_lines'] / np.maximum(1, df['unique_files_edited'])
    
    # 5. Tool intensity: tool calls per conversational turn
    df['DERIVED_tool_intensity'] = df['tool_calls_total'] / np.maximum(1, df['total_turns'])
    
    # 6. Read to edit ratio: files read vs files edited
    df['DERIVED_read_to_edit_ratio'] = df['unique_files_read'] / np.maximum(1, df['unique_files_edited'])
    
    # 7. Time per turn in seconds
    df['DERIVED_time_per_turn'] = df['duration_seconds'] / np.maximum(1, df['total_turns'])
    
    # 8. Tool entropy: information-theoretic diversity of tool sequence
    df['DERIVED_tool_entropy'] = df['tool_call_sequence'].apply(compute_tool_entropy)
    
    # 9. Log transformed duration (stabilizes high right-skewness)
    df['DERIVED_log_duration'] = np.log1p(np.maximum(0.0, df['duration_seconds']))
    
    # 10. Log transformed patch size
    df['DERIVED_log_patch_size'] = np.log1p(np.maximum(0.0, df['patch_size_lines']))
    
    return df

def split_and_preprocess(raw_parquet_path="data/raw/agent_diagnostics/signals.parquet", output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading raw dataset from {raw_parquet_path}...")
    df_raw = pd.read_parquet(raw_parquet_path)
    print(f"Loaded {len(df_raw)} records with columns: {df_raw.columns.tolist()[:10]}...")
    
    # Engineer derived features
    df = engineer_features(df_raw)
    
    # Grouped Splitting on task_id to prevent any cross-task leakage
    # 70% Train, 15% Validation, 15% Test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df['task_id']))
    
    df_train_val = df.iloc[train_val_idx].copy()
    df_test = df.iloc[test_idx].copy()
    
    # Split train_val into train (70% overall) and val (15% overall -> ~17.65% of train_val)
    val_rel_size = 0.15 / 0.85
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_rel_size, random_state=42)
    train_idx, val_idx = next(gss_val.split(df_train_val, groups=df_train_val['task_id']))
    
    df_train = df_train_val.iloc[train_idx].copy()
    df_val = df_train_val.iloc[val_idx].copy()
    
    print("\n--- Grouped Partition Statistics ---")
    print(f"Train set: {len(df_train)} samples ({len(df_train['task_id'].unique())} unique tasks), Failure rate: {df_train['is_failed'].mean():.4f}")
    print(f"Validation set: {len(df_val)} samples ({len(df_val['task_id'].unique())} unique tasks), Failure rate: {df_val['is_failed'].mean():.4f}")
    print(f"Test set: {len(df_test)} samples ({len(df_test['task_id'].unique())} unique tasks), Failure rate: {df_test['is_failed'].mean():.4f}")
    
    # Check that task overlap is strictly zero
    train_tasks = set(df_train['task_id'])
    val_tasks = set(df_val['task_id'])
    test_tasks = set(df_test['task_id'])
    assert len(train_tasks.intersection(val_tasks)) == 0, "DATA LEAKAGE: Task overlap between Train and Val!"
    assert len(train_tasks.intersection(test_tasks)) == 0, "DATA LEAKAGE: Task overlap between Train and Test!"
    assert len(val_tasks.intersection(test_tasks)) == 0, "DATA LEAKAGE: Task overlap between Val and Test!"
    print("Verification passed: Zero task-id overlap across train, val, and test splits (Strict Leakage Prevention).")
    
    # Define Numerical and Categorical feature columns
    numerical_cols = [
        'total_turns', 'tool_calls_total', 'search_tool_calls', 'edit_tool_calls',
        'code_nav_tool_calls', 'semantic_search_tool_calls', 'unique_files_read',
        'unique_files_edited', 'error_count', 'retry_count', 'patch_size_lines',
        'DERIVED_error_rate', 'DERIVED_retry_ratio', 'DERIVED_search_to_edit_ratio',
        'DERIVED_edit_churn', 'DERIVED_tool_intensity', 'DERIVED_read_to_edit_ratio',
        'DERIVED_time_per_turn', 'DERIVED_tool_entropy', 'DERIVED_log_duration',
        'DERIVED_log_patch_size'
    ]
    
    categorical_cols = ['model', 'agent_name', 'benchmark']
    
    # Fit Scaler and OneHotEncoder STRICTLY on the training set
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    print("\nFitting feature transformers strictly on Training split...")
    X_train_transformed = preprocessor.fit_transform(df_train[numerical_cols + categorical_cols])
    X_val_transformed = preprocessor.transform(df_val[numerical_cols + categorical_cols])
    X_test_transformed = preprocessor.transform(df_test[numerical_cols + categorical_cols])
    
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
    all_feature_names = numerical_cols + cat_feature_names
    print(f"Transformed feature matrix dimension: {len(all_feature_names)} features ({len(numerical_cols)} numerical + {len(cat_feature_names)} one-hot categorical).")
    
    # Save processed splits and preprocessor
    df_train.to_parquet(os.path.join(output_dir, 'train.parquet'), index=False)
    df_val.to_parquet(os.path.join(output_dir, 'val.parquet'), index=False)
    df_test.to_parquet(os.path.join(output_dir, 'test.parquet'), index=False)
    
    # Save transformed arrays
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train_transformed)
    np.save(os.path.join(output_dir, 'y_train.npy'), df_train['is_failed'].values)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val_transformed)
    np.save(os.path.join(output_dir, 'y_val.npy'), df_val['is_failed'].values)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test_transformed)
    np.save(os.path.join(output_dir, 'y_test.npy'), df_test['is_failed'].values)
    
    # Save feature names and fitted preprocessor
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.joblib'))
    joblib.dump(all_feature_names, os.path.join(output_dir, 'feature_names.joblib'))
    joblib.dump(numerical_cols, os.path.join(output_dir, 'numerical_cols.joblib'))
    joblib.dump(categorical_cols, os.path.join(output_dir, 'categorical_cols.joblib'))
    
    print(f"Successfully saved all processed splits and transformers to {output_dir}/")

if __name__ == '__main__':
    split_and_preprocess()
