"""
Audio preprocessing for MELD
- Converts audio to mono
- Normalizes sample rate
- Trims silence
"""

import librosa
import soundfile as sf
import os

TARGET_SR = 16000

def preprocess_audio(input_wav, output_wav):
    audio, sr = librosa.load(input_wav, sr=TARGET_SR, mono=True)
    audio, _ = librosa.effects.trim(audio)
    os.makedirs(os.path.dirname(output_wav), exist_ok=True)
    sf.write(output_wav, audio, TARGET_SR)

if __name__ == "__main__":
    print("Audio preprocessing module ready.")
