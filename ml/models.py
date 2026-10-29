"""
Model Architectures Module
==========================
Agent Health Intelligence (AUCML — TE7483)

Single Responsibility:
- Classical baseline classifiers (Logistic Regression, Random Forest, XGBoost)
- Sequential recurrent trajectory model (BiLSTM)
- Proposed Heterogeneous Multi-View Fusion Network
"""

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ------------------------------------------------------------------------------
# 1. CLASSICAL TABULAR BASELINE MODELS
# ------------------------------------------------------------------------------
def get_logistic_regression() -> LogisticRegression:
    """Standard linear baseline with balanced class weights."""
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs"
    )

def get_random_forest() -> RandomForestClassifier:
    """Non-linear ensemble baseline achieving highest raw accuracy (84.27%)."""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

def get_xgboost() -> XGBClassifier:
    """Gradient boosted tree baseline achieving highest Macro-F1 (0.7659)."""
    return XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        scale_pos_weight=3.34,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

# ------------------------------------------------------------------------------
# 2. NEURAL RECURRENT TRAJECTORY MODEL (BiLSTM)
# ------------------------------------------------------------------------------
class SequentialBiLSTM(nn.Module):
    """
    Recurrent sequence classifier processing tool-call trajectories.
    Input: [batch_size, seq_len]
    Output: [batch_size, 1] failure probability logit
    """
    def __init__(self, vocab_size: int = 50, embedding_dim: int = 32, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, seq_tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(seq_tokens)
        _, (h_n, _) = self.bilstm(embedded)
        # Concatenate forward and backward final hidden states
        h_cat = torch.cat([h_n[0], h_n[1]], dim=1)
        logits = self.fc(h_cat)
        return logits.squeeze(-1)

# ------------------------------------------------------------------------------
# 3. PROPOSED HETEROGENEOUS MULTI-VIEW FUSION NETWORK
# ------------------------------------------------------------------------------
class ProposedMultiViewFusion(nn.Module):
    """
    Proposed Heterogeneous Multi-View Fusion Network.
    Combines two complementary views:
      View 1 (Tabular Telemetry): Static runtime metrics, error rates, churn ratios.
      View 2 (Sequential Execution): Dynamic tool-invocation trajectory via BiLSTM.
    
    Achieves highest Failure Recall (72.95%), minimizing undetected agent failures.
    """
    def __init__(
        self,
        tabular_dim: int = 88,
        vocab_size: int = 50,
        embedding_dim: int = 32,
        bilstm_hidden_dim: int = 64,
        fusion_hidden_dim: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        # View 1: Tabular Telemetry Encoder
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # View 2: Sequential Tool-Trajectory Encoder
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=bilstm_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.seq_proj = nn.Linear(bilstm_hidden_dim * 2, 32)
        
        # Heterogeneous Cross-View Fusion Head
        self.fusion_head = nn.Sequential(
            nn.Linear(32 + 32, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1)
        )

    def forward(self, tabular_x: torch.Tensor, seq_tokens: torch.Tensor) -> torch.Tensor:
        # View 1 forward pass
        tab_feat = self.tabular_encoder(tabular_x)
        
        # View 2 forward pass
        embedded = self.embedding(seq_tokens)
        _, (h_n, _) = self.bilstm(embedded)
        seq_cat = torch.cat([h_n[0], h_n[1]], dim=1)
        seq_feat = torch.relu(self.seq_proj(seq_cat))
        
        # Cross-View Joint Fusion
        fused = torch.cat([tab_feat, seq_feat], dim=1)
        logits = self.fusion_head(fused)
        return logits.squeeze(-1)
