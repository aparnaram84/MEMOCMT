"""
Temporal Transformer for conversational emotion dynamics
"""

import torch.nn as nn

class TemporalTransformer(nn.Module):
    def __init__(self, d_model=256, layers=4, heads=8):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, x):
        return self.encoder(x)
