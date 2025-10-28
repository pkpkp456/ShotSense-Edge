# === BiLSTM-Attention Classifier (Embeddings-only) ===


import torch
import torch.nn as nn

class BiLSTM_Attention(nn.Module):
    def __init__(self, emb_dim=2048, hidden_dim=128, num_classes=2):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim*2, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        attn_weights = self.attention(lstm_out)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context)
