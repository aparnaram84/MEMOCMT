"""
Audio feature extraction using HuBERT
"""
import torch
import torchaudio
from transformers import HubertModel

model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
model.eval()

def extract_audio_features(wav_path):
    waveform, _ = torchaudio.load(wav_path)
    with torch.no_grad():
        features = model(waveform).last_hidden_state
    return features.mean(dim=1)
