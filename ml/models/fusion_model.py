"""
Proposed Heterogeneous Multi-View Trace Fusion Network (PyTorch)
"""
import torch
import torch.nn as nn

class MultimodalAgentFailureModel(nn.Module):
    """
    Heterogeneous Multi-View Architecture:
    - View 1: 88-dimensional structured telemetry & operational features
    - View 2: Sequential tool-call trajectory tokens encoded via BiLSTM
    - Fusion: Cross-modal adaptive sigmoid gating mechanism
    """
    def __init__(self, tab_dim=88, vocab_size=32, emb_dim=32, seq_hidden_dim=32, fusion_dim=64):
        super().__init__()
        # Branch 1: Structured Telemetry Deep Encoder
        self.tab_encoder = nn.Sequential(
            nn.Linear(tab_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, fusion_dim),
            nn.ReLU()
        )
        
        # Branch 2: Sequence Trajectory BiLSTM Extractor
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.bilstm = nn.LSTM(emb_dim, seq_hidden_dim, batch_first=True, bidirectional=True)
        self.seq_proj = nn.Sequential(
            nn.Linear(seq_hidden_dim * 2, fusion_dim),
            nn.ReLU()
        )
        
        # Cross-Modal Adaptive Sigmoid Gating Mechanism
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
        # Branch 1: Tabular representation
        h_tab = self.tab_encoder(tab_x)
        
        # Branch 2: Sequential tool representation
        emb = self.embedding(seq_x)
        out, _ = self.bilstm(emb)
        pooled, _ = torch.max(out, dim=1)
        h_seq = self.seq_proj(pooled)
        
        # Gated Cross-Modal Fusion
        concat = torch.cat([h_tab, h_seq], dim=-1)
        alpha = self.gate(concat)
        h_fused = alpha * h_tab + (1.0 - alpha) * h_seq
        
        logits = self.classifier(h_fused)
        return logits.squeeze(-1)
