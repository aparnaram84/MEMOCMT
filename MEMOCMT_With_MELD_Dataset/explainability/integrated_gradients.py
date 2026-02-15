"""
Integrated Gradients for TEXT explainability (correct modality handling)
"""

import torch
import matplotlib.pyplot as plt


def integrated_gradients(model, audio_input, visual_input, text_input, save_path, steps=50):
    """
    Compute Integrated Gradients for text modality.

    audio_input  : torch.Tensor (1, A)
    visual_input : torch.Tensor (1, V)
    text_input   : torch.Tensor (1, T)
    """

    model.zero_grad()

    baseline = torch.zeros_like(text_input)
    grads = []

    for alpha in torch.linspace(0, 1, steps):
        text_step = baseline + alpha * (text_input - baseline)
        text_step.requires_grad = True

        output = model(
            audio=audio_input,
            visual=visual_input,
            text=text_step
        )

        score = output.max()
        score.backward()

        grads.append(text_step.grad.detach())

    avg_grads = torch.mean(torch.stack(grads), dim=0)
    ig = (text_input - baseline) * avg_grads
    ig = ig.squeeze().cpu().numpy()

    # Plot
    plt.figure(figsize=(8, 2))
    plt.plot(ig)
    plt.title("Integrated Gradients (Text)")
    plt.xlabel("Text Feature Dimension")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
