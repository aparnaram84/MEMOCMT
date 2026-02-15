import torch.nn as nn

class CrossModalTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=8):
        super().__init__()

        self.av = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.at = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.va = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.vt = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ta = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.tv = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, audio, visual, text):
        # Add sequence dimension
        audio  = audio.unsqueeze(1)
        visual = visual.unsqueeze(1)
        text   = text.unsqueeze(1)

        # -------- Cross-modal attentions --------

        # Audio attends to Visual (STORE THIS for XAI)
        av, av_attn = self.av(
            audio,
            visual,
            visual,
            need_weights=True
        )

        # Store attention weights safely (no gradients)
        self.last_attn = av_attn.detach()

        # Other cross-modal attentions (no need to store)
        at,_ = self.at(audio, text, text)

        va,_ = self.va(visual, audio, audio)
        vt,_ = self.vt(visual, text, text)

        ta,_ = self.ta(text, audio, audio)
        tv,_ = self.tv(text, visual, visual)

        fused = av + at + va + vt + ta + tv
        fused = self.norm(fused)

        return fused.squeeze(1)
