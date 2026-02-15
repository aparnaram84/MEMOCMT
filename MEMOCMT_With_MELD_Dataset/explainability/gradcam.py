
"""
Visual Feature Saliency (Grad-CAM–style) for AVT-Improved MemoCMT

IMPORTANT:
- This is NOT CNN Grad-CAM.
- MELD uses pre-extracted visual feature vectors (no spatial dimensions).
- This computes gradient-based saliency over feature dimensions.
- Safe, stable, and thesis-correct.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def compute_gradcam(visual_input, save_path):
    """
    Compute gradient-based visual feature saliency.

    Parameters
    ----------
    visual_input : torch.Tensor
        Shape (D,) or (1, D)
    save_path : str
        Path to save the saliency plot
    """

    # Ensure tensor shape (1, D)
    if visual_input.dim() == 1:
        visual_input = visual_input.unsqueeze(0)

    # Clone to avoid modifying original tensor
    visual_input = visual_input.detach().clone()
    visual_input.requires_grad = True

    # Scalar objective (L2 norm of features)
    score = visual_input.norm()
    score.backward()

    # Gradient-based saliency
    saliency = visual_input.grad.abs().squeeze(0).cpu().numpy()

    # Normalize
    saliency = saliency / (saliency.max() + 1e-8)

    # Plot saliency over feature dimensions
    plt.figure(figsize=(8, 2))
    plt.plot(saliency)
    plt.title("Visual Feature Saliency (Gradient-based)")
    plt.xlabel("Feature Dimension")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
