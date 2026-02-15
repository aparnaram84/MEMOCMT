import torch.nn as nn

class ModalityEncoders(nn.Module):
    """
    Projects modality-specific features into a shared latent space.
    """

    def __init__(self, d_model=256):
        super().__init__()

        self.audio_proj  = nn.Linear(768, d_model)
        self.visual_proj = nn.Linear(2048, d_model)
        self.text_proj   = nn.Linear(768, d_model)

    def forward(self, audio, visual, text):
        audio  = self.audio_proj(audio)
        visual = self.visual_proj(visual)
        text   = self.text_proj(text)
        return audio, visual, text
