import pandas as pd
import json
import os

print("Loading datasets...")
train = pd.read_parquet("data/processed/train.parquet")
val = pd.read_parquet("data/processed/val.parquet")
test = pd.read_parquet("data/processed/test.parquet")

train["split"] = "train"
val["split"] = "val"
test["split"] = "test"

full = pd.concat([train, val, test], ignore_index=True)

# Convert list columns to clean string representation for Excel compatibility
for col in ["files_read_list", "files_edited_list", "tool_call_sequence"]:
    if col in full.columns:
        full[col] = full[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))

csv_path = "data/dataset_full.csv"
print(f"Writing {csv_path} ({len(full)} rows)...")
full.to_csv(csv_path, index=False)

print("Loading test_predictions and feature dictionary...")
preds = pd.read_parquet("data/processed/test_predictions.parquet")
feat_dict = pd.read_csv("FEATURE_DICTIONARY.csv")

fail_pct = full["is_failed"].mean() * 100
succ_pct = (1.0 - full["is_failed"].mean()) * 100

overview_data = [
    {"Metadata Attribute": "Project Title", "Specification": "Multimodal AI for Agentic System Failure Diagnosis & Root-Cause Reasoning"},
    {"Metadata Attribute": "Subject Code & Rubric", "Specification": "AUCML — TE7483 (Symbiosis Institute of Technology)"},
    {"Metadata Attribute": "Total Records", "Specification": str(len(full))},
    {"Metadata Attribute": "Training Split Size", "Specification": f"{len(train)} rows (69.2%)"},
    {"Metadata Attribute": "Validation Split Size", "Specification": f"{len(val)} rows (14.2%)"},
    {"Metadata Attribute": "Test Split Size", "Specification": f"{len(test)} rows (16.6%)"},
    {"Metadata Attribute": "Total Features", "Specification": f"{len(full.columns) - 1} features"},
    {"Metadata Attribute": "Target Variable", "Specification": "is_failed (0: Nominal Run, 1: Failure Detected)"},
    {"Metadata Attribute": "Class Distribution", "Specification": f"Failure: {fail_pct:.1f}% | Success: {succ_pct:.1f}%"},
    {"Metadata Attribute": "Agent Frameworks", "Specification": "OpenHands, SWE-Agent, Aider, Modular ReAct"},
    {"Metadata Attribute": "Foundation LLMs", "Specification": "Claude-3.5-Sonnet, GPT-4o, Haiku, Llama-3-70B, Qwen-2.5-Coder"},
    {"Metadata Attribute": "Data Sources", "Specification": "TraceLabs, ToolFailBench, SWE-bench Verified"}
]
df_overview = pd.DataFrame(overview_data)

excel_path = "data/dataset_multimodal_agent_failures.xlsx"
print(f"Writing multi-sheet Excel workbook to {excel_path}...")

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    df_overview.to_excel(writer, sheet_name="Dataset_Overview", index=False)
    feat_dict.to_excel(writer, sheet_name="Feature_Dictionary", index=False)
    full.to_excel(writer, sheet_name="Complete_Dataset", index=False)
    preds.to_excel(writer, sheet_name="Test_Set_Predictions", index=False)

print(f"Successfully created: {excel_path} ({os.path.getsize(excel_path) / (1024*1024):.2f} MB)")
