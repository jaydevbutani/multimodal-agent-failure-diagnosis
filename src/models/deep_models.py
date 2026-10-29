import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

# Set seeds for complete reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Build Vocabulary from Training Set Only
def build_vocab(sequences):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for seq in sequences:
        if seq is not None:
            for token in seq:
                token_str = str(token)
                if token_str not in vocab:
                    vocab[token_str] = len(vocab)
    return vocab

def encode_sequence(seq, vocab, max_len=50):
    if seq is None or len(seq) == 0:
        return [0] * max_len
    encoded = [vocab.get(str(t), 1) for t in seq[:max_len]]
    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))
    return encoded

# 2. PyTorch Dataset
class AgentTelemetryDataset(Dataset):
    def __init__(self, tabular_X, sequences, labels, vocab, max_len=50):
        self.tabular_X = torch.tensor(tabular_X, dtype=torch.float32)
        encoded_seqs = [encode_sequence(s, vocab, max_len) for s in sequences]
        self.seq_X = torch.tensor(encoded_seqs, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.tabular_X[idx], self.seq_X[idx], self.labels[idx]

# 3. Model 3: Trajectory Sequence Model (BiLSTM)
class SequenceTrajectoryModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.bilstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
    def forward(self, seq_x):
        # seq_x: [batch, max_len]
        emb = self.embedding(seq_x)  # [batch, max_len, emb_dim]
        out, (hn, _) = self.bilstm(emb)
        # Max-pool over sequence length
        pooled, _ = torch.max(out, dim=1)  # [batch, hidden_dim * 2]
        logits = self.classifier(self.dropout(pooled))
        return logits.squeeze(-1)

# 4. Model 4: Proposed Multimodal Cross-Modal Attention Fusion Network
class MultimodalAgentFailureModel(nn.Module):
    def __init__(self, tab_dim, vocab_size, emb_dim=32, seq_hidden_dim=32, fusion_dim=64):
        super().__init__()
        # Branch 1: Telemetry Feature Extractor
        self.tab_encoder = nn.Sequential(
            nn.Linear(tab_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, fusion_dim),
            nn.ReLU()
        )
        
        # Branch 2: Sequence Trajectory Extractor
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.bilstm = nn.LSTM(emb_dim, seq_hidden_dim, batch_first=True, bidirectional=True)
        self.seq_proj = nn.Sequential(
            nn.Linear(seq_hidden_dim * 2, fusion_dim),
            nn.ReLU()
        )
        
        # Cross-Modal Gated Fusion Mechanism
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid()
        )
        
        # Final Diagnosis Head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
    def forward(self, tab_x, seq_x):
        # Branch 1
        h_tab = self.tab_encoder(tab_x)  # [batch, fusion_dim]
        
        # Branch 2
        emb = self.embedding(seq_x)
        out, _ = self.bilstm(emb)
        pooled, _ = torch.max(out, dim=1)
        h_seq = self.seq_proj(pooled)  # [batch, fusion_dim]
        
        # Gated Fusion: learns dynamically how much weight to place on telemetry vs sequence
        concat = torch.cat([h_tab, h_seq], dim=-1)
        alpha = self.gate(concat)
        h_fused = alpha * h_tab + (1.0 - alpha) * h_seq
        
        logits = self.classifier(h_fused)
        return logits.squeeze(-1)

# 5. Training and Evaluation Routine
def train_and_eval_model(model, train_loader, val_loader, test_loader, epochs=15, is_multimodal=True, lr=0.001):
    pos_weight = torch.tensor([3.34])  # Class imbalance correction: 76.9% Pass vs 23.1% Fail
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_f1 = 0.0
    best_weights = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for tab_x, seq_x, y in train_loader:
            optimizer.zero_grad()
            if is_multimodal:
                preds = model(tab_x, seq_x)
            else:
                preds = model(seq_x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        val_preds_list, val_y_list = [], []
        with torch.no_grad():
            for tab_x, seq_x, y in val_loader:
                if is_multimodal:
                    out = model(tab_x, seq_x)
                else:
                    out = model(seq_x)
                probs = torch.sigmoid(out).numpy()
                val_preds_list.extend(probs)
                val_y_list.extend(y.numpy())
                
        val_preds = (np.array(val_preds_list) >= 0.5).astype(int)
        val_f1 = f1_score(val_y_list, val_preds, average='macro', zero_division=0)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = model.state_dict().copy()
            
    # Load best weights for final test evaluation
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    model.eval()
    test_probs_list, test_y_list = [], []
    with torch.no_grad():
        for tab_x, seq_x, y in test_loader:
            if is_multimodal:
                out = model(tab_x, seq_x)
            else:
                out = model(seq_x)
            probs = torch.sigmoid(out).numpy()
            test_probs_list.extend(probs)
            test_y_list.extend(y.numpy())
            
    test_probs = np.array(test_probs_list)
    test_y = np.array(test_y_list)
    test_preds = (test_probs >= 0.5).astype(int)
    
    acc = accuracy_score(test_y, test_preds)
    prec = precision_score(test_y, test_preds, zero_division=0)
    rec = recall_score(test_y, test_preds, zero_division=0)
    macro_f1 = f1_score(test_y, test_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(test_y, test_preds, average='weighted', zero_division=0)
    roc_auc = roc_auc_score(test_y, test_probs)
    pr_auc = average_precision_score(test_y, test_probs)
    cm = confusion_matrix(test_y, test_preds).tolist()
    
    return {
        'accuracy': round(float(acc), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'macro_f1': round(float(macro_f1), 4),
        'weighted_f1': round(float(weighted_f1), 4),
        'roc_auc': round(float(roc_auc), 4),
        'pr_auc': round(float(pr_auc), 4),
        'confusion_matrix': cm
    }

def run_deep_experiments(data_dir="data/processed", results_dir="results"):
    df_train = pd.read_parquet(os.path.join(data_dir, "train.parquet"))
    df_val = pd.read_parquet(os.path.join(data_dir, "val.parquet"))
    df_test = pd.read_parquet(os.path.join(data_dir, "test.parquet"))
    
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    vocab = build_vocab(df_train['tool_call_sequence'])
    print(f"Tool Vocabulary size: {len(vocab)} unique tool tokens.")
    joblib.dump(vocab, os.path.join(data_dir, "tool_vocab.joblib"))
    
    train_ds = AgentTelemetryDataset(X_train, df_train['tool_call_sequence'], y_train, vocab)
    val_ds = AgentTelemetryDataset(X_val, df_val['tool_call_sequence'], y_val, vocab)
    test_ds = AgentTelemetryDataset(X_test, df_test['tool_call_sequence'], y_test, vocab)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    
    # Model 3: Sequence Model Only
    print("\n--- Training Model 3: Sequence Trajectory Model (BiLSTM) ---")
    model_seq = SequenceTrajectoryModel(vocab_size=len(vocab))
    seq_results = train_and_eval_model(model_seq, train_loader, val_loader, test_loader, epochs=12, is_multimodal=False)
    seq_results['model'] = 'Sequential BiLSTM (Trajectory Only)'
    print("Sequential BiLSTM Test Results:", seq_results)
    
    # Model 4: Proposed Multimodal Cross-Modal Fusion
    print("\n--- Training Model 4: Proposed Multimodal Cross-Attention Fusion Network ---")
    model_fusion = MultimodalAgentFailureModel(tab_dim=X_train.shape[1], vocab_size=len(vocab))
    fusion_results = train_and_eval_model(model_fusion, train_loader, val_loader, test_loader, epochs=15, is_multimodal=True)
    fusion_results['model'] = 'Proposed Multimodal Fusion Network'
    print("Proposed Multimodal Fusion Test Results:", fusion_results)
    
    # Save Model Weights
    torch.save(model_fusion.state_dict(), "models_saved/multimodal_fusion.pt")
    torch.save(model_seq.state_dict(), "models_saved/sequence_bilstm.pt")
    
    # Save Confusion Matrix Plots
    for res in [seq_results, fusion_results]:
        cm = np.array(res['confusion_matrix'])
        plt.figure(figsize=(4.5, 3.8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=['Normal (0)', 'Failed (1)'],
                    yticklabels=['Normal (0)', 'Failed (1)'])
        plt.title(f"Confusion Matrix - {res['model']}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Ground Truth")
        plt.tight_layout()
        safe_name = res['model'].lower().replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(os.path.join(results_dir, "plots", f"cm_{safe_name}.png"), dpi=200)
        plt.close()
        
    # Save metrics
    with open(os.path.join(results_dir, "metrics", "deep_test_metrics.json"), "w") as f:
        json.dump([seq_results, fusion_results], f, indent=2)
        
    print("\nDeep sequence and multimodal models successfully trained and evaluated.")
    return [seq_results, fusion_results]

if __name__ == '__main__':
    run_deep_experiments()
