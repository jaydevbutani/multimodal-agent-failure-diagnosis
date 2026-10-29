import json
import os
import pandas as pd
import numpy as np

def profile_toolfailbench():
    print("="*70)
    print("PROFILING DATASET 1: ToolFailBench (Execution Failure Diagnosis)")
    print("="*70)
    
    files = ['data/raw/toolfailbench_claude_haiku.json', 'data/raw/toolfailbench_gpt54_mini.json']
    all_records = []
    for f in files:
        with open(f, 'r', encoding='utf-8') as fp:
            records = json.load(fp)
            all_records.extend(records)
            print(f"Loaded {len(records)} records from {os.path.basename(f)}")
            
    print(f"Total ToolFailBench records: {len(all_records)}")
    
    # Analyze fields
    df = pd.DataFrame([{
        'task_id': r['task']['task_id'],
        'domain': r['task']['domain'],
        'turn_type': r['task']['turn_type'],
        'target_failure_mode': r['task']['target_failure_mode'],
        'conflict_type': r['task'].get('conflict_type', 'none'),
        'model_id': r.get('model_id', 'unknown'),
        'classification': r.get('classification', 'unknown'),
        'has_tool_call': len(r.get('agent_trace', {}).get('tool_calls', [])) > 0,
        'num_tool_calls': len(r.get('agent_trace', {}).get('tool_calls', [])),
        'user_msg_len': len(r['task']['user_message']),
        'agent_ans_len': len(r.get('agent_answer', '')),
        'is_failed': r.get('classification') != 'correct'
    } for r in all_records])
    
    print("\nDataFrame Shape:", df.shape)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDomain Distribution:\n", df['domain'].value_counts())
    print("\nBinary Target (is_failed):\n", df['is_failed'].value_counts(normalize=True).round(4))
    print("\nFailure Classification Target:\n", df['classification'].value_counts())
    print("\nCross-tabulation (Domain vs Classification):\n", pd.crosstab(df['domain'], df['classification']))
    print("\nTelemetry Features summary (num_tool_calls, lengths):")
    print(df[['num_tool_calls', 'user_msg_len', 'agent_ans_len']].describe())

def profile_who_and_when():
    print("\n" + "="*70)
    print("PROFILING DATASET 2: Who&When (ICML 2025 Multi-Agent Failure Attribution)")
    print("="*70)
    
    df_ag = pd.read_parquet('data/raw/who_and_when_ag.parquet')
    df_hc = pd.read_parquet('data/raw/who_and_when_hc.parquet')
    
    print(f"Algorithm-Generated traces: {len(df_ag)}")
    print(f"Hand-Crafted traces: {len(df_hc)}")
    
    # Analyze step distributions
    step_rows = []
    for idx, row in df_hc.iterrows():
        history = row['history']
        m_step = int(row['mistake_step'])
        m_agent = str(row['mistake_agent'])
        for s_idx, turn in enumerate(history):
            role = turn.get('name', turn.get('role', 'unknown'))
            content = str(turn.get('content', ''))
            is_critical_failure_step = (s_idx == m_step)
            step_phase = 'pre_failure' if s_idx < m_step else ('failure_point' if s_idx == m_step else 'post_failure')
            step_rows.append({
                'trace_id': row['question_ID'],
                'step_idx': s_idx,
                'total_steps': len(history),
                'normalized_pos': s_idx / max(1, len(history)-1),
                'role': role,
                'content_len': len(content),
                'has_code': '```' in content,
                'mistake_agent': m_agent,
                'is_critical_failure': is_critical_failure_step,
                'step_phase': step_phase
            })
            
    step_df = pd.DataFrame(step_rows)
    print(f"\nHand-crafted total steps: {len(step_df)}")
    print("Step phase distribution:\n", step_df['step_phase'].value_counts())
    print("\nUnique Agents/Roles in Traces:\n", step_df['role'].value_counts())
    print("\nResponsible Failing Agent Ground Truth:\n", df_hc['mistake_agent'].value_counts())

if __name__ == '__main__':
    profile_toolfailbench()
    profile_who_and_when()
