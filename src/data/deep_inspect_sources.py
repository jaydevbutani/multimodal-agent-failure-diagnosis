import requests
import json
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
headers = {"User-Agent": "Mozilla/5.0"}

def inspect_agent_diagnostics_data():
    print("=== INSPECTING AGENT DIAGNOSTICS EXPORT ===")
    url = "https://api.github.com/repos/sjarmak/agent-diagnostics/contents/data/export"
    r = requests.get(url, headers=headers)
    print("Status:", r.status_code)
    if r.status_code == 200:
        for f in r.json():
            print(f"  {f['name']} ({f.get('size', 0)/1024:.1f} KB)")
            
def inspect_tracelab():
    print("\n=== INSPECTING TRACELAB RELEASES & ASSETS ===")
    url = "https://api.github.com/repos/uw-syfi/TraceLab/releases"
    r = requests.get(url, headers=headers)
    print("Releases status:", r.status_code)
    if r.status_code == 200:
        releases = r.json()
        print(f"Found {len(releases)} releases:")
        for rel in releases[:5]:
            print(f"  Tag: {rel.get('tag_name')}, Name: {rel.get('name')}")
            for asset in rel.get('assets', []):
                print(f"    Asset: {asset['name']} ({asset['size']/(1024*1024):.2f} MB) -> {asset['browser_download_url']}")
                
    # Also check huggingface for tracelab
    r_hf = requests.get("https://huggingface.co/api/datasets?search=tracelab", headers=headers)
    if r_hf.status_code == 200:
        print("HF datasets with tracelab:", [d['id'] for d in r_hf.json()[:5]])

def inspect_cmu_files():
    print("\n=== INSPECTING CMU AGENT TRAJECTORIES ===")
    url = "https://huggingface.co/api/datasets/cx-cmu/agent_trajectories"
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        info = r.json()
        for sib in info.get('siblings', []):
            fname = sib['rfilename']
            # Get actual file size via resolve header
            head_url = f"https://huggingface.co/datasets/cx-cmu/agent_trajectories/resolve/main/{fname}"
            r_h = requests.head(head_url, headers=headers, allow_redirects=True)
            sz = int(r_h.headers.get('content-length', 0))
            print(f"  {fname}: {sz/(1024*1024):.2f} MB")

def inspect_freeinference():
    print("\n=== INSPECTING FREEINFERENCE REPO ===")
    url = "https://api.github.com/repos/HarvardMadSys/freeinference_agentic_trace/contents"
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        for f in r.json():
            print(f"  {f['name']} ({f.get('size', 0)} bytes)")
    readme_url = "https://raw.githubusercontent.com/HarvardMadSys/freeinference_agentic_trace/main/README.md"
    r_rm = requests.get(readme_url, headers=headers)
    if r_rm.status_code == 200:
        for l in r_rm.text.split("\n")[:15]:
            print("   ", l)

if __name__ == '__main__':
    inspect_agent_diagnostics_data()
    inspect_tracelab()
    inspect_cmu_files()
    inspect_freeinference()
