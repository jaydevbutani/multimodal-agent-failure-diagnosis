import os
import sys
import io
import json
import requests
import pandas as pd

# Fix stdout encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def audit_tracelab():
    print("="*60)
    print("1. AUDITING TRACELAB (uw-syfi/TraceLab)")
    print("="*60)
    url = "https://api.github.com/repos/uw-syfi/TraceLab/contents"
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        items = r.json()
        print("Root contents:", [i['name'] for i in items])
        # Check README.md
        r_readme = requests.get("https://raw.githubusercontent.com/uw-syfi/TraceLab/main/README.md", headers=headers)
        if r_readme.status_code == 200:
            lines = r_readme.text.split("\n")
            print("README snippet:")
            for l in lines[:25]:
                if l.strip():
                    print(" ", l)

def audit_agent_diagnostics():
    print("\n" + "="*60)
    print("2. AUDITING AGENT DIAGNOSTICS (sjarmak/agent-diagnostics)")
    print("="*60)
    url = "https://api.github.com/repos/sjarmak/agent-diagnostics/contents"
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        items = r.json()
        print("Root contents:", [i['name'] for i in items])
        # Check if there is data/ or datasets/ directory
        for item in items:
            if item['name'] in ['data', 'dataset', 'datasets', 'benchmarks']:
                r_sub = requests.get(item['url'], headers=headers)
                if r_sub.status_code == 200:
                    print(f"  Contents of {item['name']}:", [s['name'] for s in r_sub.json()[:10]])
        # Check README.md snippet
        r_readme = requests.get("https://raw.githubusercontent.com/sjarmak/agent-diagnostics/main/README.md", headers=headers)
        if r_readme.status_code == 200:
            lines = r_readme.text.split("\n")
            print("README snippet:")
            for l in lines[:30]:
                if l.strip():
                    print(" ", l)

def audit_cmu_trajectories():
    print("\n" + "="*60)
    print("3. AUDITING CMU AGENT TRAJECTORIES (cx-cmu/agent_trajectories)")
    print("="*60)
    # Check README or download small preview of parquet
    r_readme = requests.get("https://huggingface.co/datasets/cx-cmu/agent_trajectories/raw/main/README.md", headers=headers)
    if r_readme.status_code == 200:
        lines = r_readme.text.split("\n")
        print("README summary lines:")
        for l in lines[:35]:
            if any(k in l.lower() for k in ['benchmark', 'records', 'passes', 'schema', 'reward', 'turns', 'overview', 'columns']):
                print(" ", l.strip())
                
    # Download header/info of search.parquet or mathhay.parquet or mcpbench.parquet
    for subset in ['mathhay', 'mcpbench', 'search']:
        url = f"https://huggingface.co/datasets/cx-cmu/agent_trajectories/resolve/main/{subset}.parquet"
        r_head = requests.head(url, headers=headers, allow_redirects=True)
        size_mb = int(r_head.headers.get('content-length', 0)) / (1024*1024)
        print(f"  {subset}.parquet: {size_mb:.2f} MB")

def audit_trace_commons():
    print("\n" + "="*60)
    print("4. AUDITING TRACE COMMONS (trace-commons/agent-traces)")
    print("="*60)
    url = "https://huggingface.co/datasets/trace-commons/agent-traces/resolve/main/data/train-00000-of-00001.parquet"
    r_head = requests.head(url, headers=headers, allow_redirects=True)
    size_mb = int(r_head.headers.get('content-length', 0)) / (1024*1024)
    print(f"train-00000-of-00001.parquet: {size_mb:.2f} MB")

def audit_freeinference():
    print("\n" + "="*60)
    print("5. AUDITING FREEINFERENCE (HarvardMadSys/freeinference_agentic_trace)")
    print("="*60)
    url = "https://api.github.com/repos/HarvardMadSys/freeinference_agentic_trace/contents"
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        items = r.json()
        print("Root contents:", [i['name'] for i in items])
        r_readme = requests.get("https://raw.githubusercontent.com/HarvardMadSys/freeinference_agentic_trace/main/README.md", headers=headers)
        if r_readme.status_code == 200:
            lines = r_readme.text.split("\n")
            print("README snippet:")
            for l in lines[:20]:
                if l.strip():
                    print(" ", l)

if __name__ == '__main__':
    audit_tracelab()
    audit_agent_diagnostics()
    audit_cmu_trajectories()
    audit_trace_commons()
    audit_freeinference()
