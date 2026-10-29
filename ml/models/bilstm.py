"""
Sequential Trajectory BiLSTM Model Definition (PyTorch)
"""
import torch
import torch.nn as nn

class SequenceTrajectoryModel(nn.Module):
    """
    Bidirectional LSTM trajectory encoder over discrete agent tool invocations.
    Pools token sequence representations to classify agent health status.
    """
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
        # seq_x: [batch_size, max_len]
        emb = self.embedding(seq_x)  # [batch_size, max_len, emb_dim]
        out, _ = self.bilstm(emb)
        # Max-pooling across time dimensions
        pooled, _ = torch.max(out, dim=1)  # [batch_size, hidden_dim * 2]
        logits = self.classifier(self.dropout(pooled))
        return logits.squeeze(-1)
