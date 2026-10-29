import requests
import json

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def check_github_repo(owner_repo):
    url = f"https://api.github.com/repos/{owner_repo}"
    r = requests.get(url, headers=headers, timeout=15)
    if r.status_code == 200:
        data = r.json()
        print(f"[GitHub] {owner_repo}:")
        print(f"  Description: {data.get('description')}")
        print(f"  Stars: {data.get('stargazers_count')}, Forks: {data.get('forks_count')}")
        print(f"  License: {data.get('license', {}).get('name') if data.get('license') else 'None'}")
        
        # Check contents
        contents_url = f"https://api.github.com/repos/{owner_repo}/contents"
        r_c = requests.get(contents_url, headers=headers, timeout=15)
        if r_c.status_code == 200:
            items = [item['name'] for item in r_c.json()[:15]]
            print(f"  Top contents: {items}")
        return data
    else:
        print(f"[GitHub] {owner_repo} failed with status {r.status_code}")
        return None

def check_hf_dataset(dataset_id):
    url = f"https://huggingface.co/api/datasets/{dataset_id}"
    r = requests.get(url, headers=headers, timeout=15)
    if r.status_code == 200:
        data = r.json()
        print(f"[HuggingFace] {dataset_id}:")
        print(f"  Description: {data.get('description', '')[:150]}")
        siblings = [s['rfilename'] for s in data.get('siblings', [])[:15]]
        print(f"  Files: {siblings}")
        return data
    else:
        print(f"[HuggingFace] {dataset_id} failed with status {r.status_code}")
        return None

if __name__ == '__main__':
    print("=== INVESTIGATING CANDIDATE DATASETS ===")
    
    print("\n--- 1. TraceLab (uw-syfi/TraceLab) ---")
    check_github_repo("uw-syfi/TraceLab")
    
    print("\n--- 2. Agent Diagnostics (sjarmak/agent-diagnostics) ---")
    check_github_repo("sjarmak/agent-diagnostics")
    
    print("\n--- 3. CMU Agent Trajectories (cx-cmu/agent_trajectories) ---")
    check_hf_dataset("cx-cmu/agent_trajectories")
    
    print("\n--- 4. Trace Commons (trace-commons/agent-traces) ---")
    check_hf_dataset("trace-commons/agent-traces")
    
    print("\n--- 5. FreeInference (HarvardMadSys/freeinference_agentic_trace) ---")
    check_github_repo("HarvardMadSys/freeinference_agentic_trace")
    
    print("\n--- 6. Who&When (ag2ai/Agents_Failure_Attribution) ---")
    check_github_repo("ag2ai/Agents_Failure_Attribution")
