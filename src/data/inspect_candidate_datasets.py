import os
import json
import pandas as pd
import requests

def inspect_who_and_when():
    print("="*60)
    print("ANALYZING CANDIDATE 1: Who&When (ICML 2025 Benchmark)")
    print("="*60)
    df_ag = pd.read_parquet('data/raw/who_and_when_ag.parquet')
    df_hc = pd.read_parquet('data/raw/who_and_when_hc.parquet')
    
    print(f"Algorithm-Generated traces: {len(df_ag)}")
    print(f"Hand-Crafted traces: {len(df_hc)}")
    total_traces = len(df_ag) + len(df_hc)
    print(f"Total traces: {total_traces}")
    
    # Analyze total steps across all traces
    ag_steps = sum(len(h) for h in df_ag['history'])
    hc_steps = sum(len(h) for h in df_hc['history'])
    print(f"Total execution steps/turns: {ag_steps + hc_steps} (AG: {ag_steps}, HC: {hc_steps})")
    print(f"Average steps per trace: {(ag_steps + hc_steps)/total_traces:.1f}")
    
    # Hand-crafted component distribution
    print("\nHand-Crafted Components (mistake_agent):")
    print(df_hc['mistake_agent'].value_counts())
    
    # Algorithm-generated component clusters
    print("\nAlgorithm-Generated Components (Top 10):")
    print(df_ag['mistake_agent'].value_counts().head(10))
    
    # Mistake step distribution
    print("\nDecisive Error Step (Hand-Crafted):")
    print(df_hc['mistake_step'].value_counts().head(5))

def inspect_mad():
    print("\n" + "="*60)
    print("ANALYZING CANDIDATE 2: MAD (MAST Taxonomy)")
    print("="*60)
    with open('data/raw/mad_human_labelled.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Human-annotated traces in file: {len(data)}")
    if len(data) > 0:
        sample = data[0]
        print("Trace keys:", list(sample.keys()))
        print("Number of annotations:", len(sample['annotations']))

def inspect_toolfailbench():
    print("\n" + "="*60)
    print("ANALYZING CANDIDATE 3: ToolFailBench Traces")
    print("="*60)
    url = "https://huggingface.co/datasets/SoHarshh/toolfailbench-traces/resolve/main/claude-haiku-4-5_20260505_182847.json"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, stream=True)
        chunk = b""
        for c in r.iter_content(chunk_size=16384):
            chunk += c
            if len(chunk) > 200000:
                break
        text = chunk.decode("utf-8", "ignore")
        # Extract first 50 classifications
        import re
        classes = re.findall(r'"classification":\s*"([^"]+)"', text)
        print(f"Extracted classifications sample ({len(classes)} items):")
        from collections import Counter
        print(Counter(classes))
    except Exception as e:
        print("Error inspecting ToolFailBench:", e)

if __name__ == '__main__':
    inspect_who_and_when()
    inspect_mad()
    inspect_toolfailbench()
