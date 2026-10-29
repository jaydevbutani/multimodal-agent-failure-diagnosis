import pandas as pd
import json

df_ag = pd.read_parquet('data/raw/who_and_when_ag.parquet')
df_hc = pd.read_parquet('data/raw/who_and_when_hc.parquet')

print("="*60)
print("1. WHO & WHEN: ALGORITHM-GENERATED (CaptainAgent)")
print("="*60)
print("Shape:", df_ag.shape)
print("Columns:", df_ag.columns.tolist())
print("\nMissing values:\n", df_ag.isnull().sum())
print("\nis_correct value counts:\n", df_ag['is_correct'].value_counts(dropna=False))
print("\nResponsible Agent (mistake_agent) value counts:\n", df_ag['mistake_agent'].value_counts(dropna=False))
print("\nDecisive Error Step (mistake_step) summary:\n", df_ag['mistake_step'].describe())

print("\n" + "="*60)
print("2. WHO & WHEN: HAND-CRAFTED (Magentic-One)")
print("="*60)
print("Shape:", df_hc.shape)
print("Columns:", df_hc.columns.tolist())
print("\nMissing values:\n", df_hc.isnull().sum())
print("\nis_corrected / is_correct value counts:\n", df_hc['is_corrected'].value_counts(dropna=False))
print("\nResponsible Agent (mistake_agent) value counts:\n", df_hc['mistake_agent'].value_counts(dropna=False))
print("\nMistake Type (mistake_type) value counts:\n", df_hc['mistake_type'].value_counts(dropna=False))
print("\nDecisive Error Step (mistake_step) summary:\n", df_hc['mistake_step'].describe())

print("\n" + "="*60)
print("3. MAD HUMAN LABELLED DATASET (MAST Taxonomy)")
print("="*60)
with open('data/raw/mad_human_labelled.json', 'r', encoding='utf-8') as f:
    mad_data = json.load(f)
print("Total records:", len(mad_data))
if len(mad_data) > 0:
    print("MAS systems:", set(d.get('mas_name') for d in mad_data))
    print("Benchmarks:", set(d.get('benchmark_name') for d in mad_data))
    print("Annotations count per trace:", len(mad_data[0].get('annotations', [])))
    print("Annotation failure categories:")
    for a in mad_data[0].get('annotations', [])[:5]:
        print("  -", a.get('failure mode').split('\n')[0])

print("\n" + "="*60)
print("4. DEEP INSPECTION OF WHO & WHEN TRAJECTORIES")
print("="*60)
sample = df_ag.iloc[0]
print("Task Question:", sample['question'][:120], "...")
print("Ground truth answer:", sample['ground_truth'])
print("Mistake Agent:", sample['mistake_agent'])
print("Mistake Step:", sample['mistake_step'])
print("Mistake Reason:", sample['mistake_reason'])
print("Trajectory length (turns):", len(sample['history']))
for i, turn in enumerate(sample['history'][:4]):
    print(f"  Step {i}: keys={list(turn.keys())}, role={turn.get('name', turn.get('role', 'unknown'))}")
    content = str(turn.get('content', ''))
    print(f"    Content snippet: {content[:100]}...")
