"""
Robust cross-modal attention visualization
Handles utterance-level (1x1) and general cases safely
"""

import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_attention(attention_weights, save_path):
    """
    attention_weights: torch.Tensor of shape (B, H, Q, K)
    """

    if not isinstance(attention_weights, torch.Tensor):
        raise ValueError("attention_weights must be a torch.Tensor")

    # ---- Reduce tensor safely ----
    attn = attention_weights.detach().cpu()

    # Remove batch dimension
    if attn.dim() == 4:
        attn = attn.squeeze(0)   # (H, Q, K)

    # Average over heads
    if attn.dim() == 3:
        attn = attn.mean(dim=0)  # (Q, K)

    # Convert to numpy
    attn = attn.numpy()

    plt.figure(figsize=(4, 4))

    # ---- UTTERANCE-LEVEL CASE (Q*K <= 1) ----
    if attn.ndim == 1 or attn.size <= 1:
        value = float(attn.reshape(-1)[0])
        plt.bar(["Audio → Visual"], [value])
        plt.ylim(0, 1)
        plt.ylabel("Attention Weight")
        plt.title("Cross-Modal Attention Strength")

    # ---- GENERAL CASE (SAFE HEATMAP WITHOUT SEABORN) ----
    else:
        plt.imshow(attn, cmap="Blues", aspect="auto")
        plt.colorbar(label="Attention Weight")
        plt.title("Audio–Visual Cross-Modal Attention")
        plt.xlabel("Visual")
        plt.ylabel("Audio")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
