import torch
import torch.nn as nn

from models.encoders import ModalityEncoders
from models.cross_modal_transformer import CrossModalTransformer
from models.temporal_transformer import TemporalTransformer

class AVT_MemoCMT(nn.Module):
    """
    Audio-Visual-Text MemoCMT model

    - Uses modality-specific projection layers via ModalityEncoders
    - Applies cross-modal attention
    - Applies temporal transformer
    - Outputs emotion class logits
    """

    def __init__(self, d_model=256, num_classes=7):
        super().__init__()

        # ✅ Modality-specific projection (FIXED)
        self.encoders = ModalityEncoders(d_model=d_model)

        # Cross-modal transformer
        self.cmt = CrossModalTransformer(d_model=d_model)

        # Temporal transformer
        self.temporal = TemporalTransformer(d_model=d_model)

        # Final classifier
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, audio, visual, text):
        """
        Inputs:
            audio  : Tensor (768,)
            visual : Tensor (2048,)
            text   : Tensor (768,)

        Output:
            logits : Tensor (num_classes,)
        """

        # 1️⃣ Project to common latent space (256-D)
        audio, visual, text = self.encoders(audio, visual, text)

        # 2️⃣ Cross-modal fusion
        fused = self.cmt(audio, visual, text)  # (256,)

        # 3️⃣ Temporal modeling
        # Add fake time dimension (utterance-level safe)
        fused = fused.unsqueeze(1)              # (B=1, T=1, D)
        fused = self.temporal(fused)
        fused = fused.squeeze(1)                # (256,)

        # 4️⃣ Classification
        logits = self.classifier(fused)

        return logits
