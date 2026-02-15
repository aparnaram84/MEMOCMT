"""
Audio preprocessing with Voice Activity Detection (VAD)
- Resamples audio
- Removes non-speech segments
"""

import librosa
import soundfile as sf
import webrtcvad
import numpy as np
import os

vad = webrtcvad.Vad(2)
TARGET_SR = 16000

def preprocess_audio(input_wav, output_wav):
    audio, sr = librosa.load(input_wav, sr=TARGET_SR, mono=True)
    audio_int16 = (audio * 32768).astype(np.int16)

    frame_duration = 30  # ms
    frame_length = int(TARGET_SR * frame_duration / 1000)

    voiced = []
    for i in range(0, len(audio_int16), frame_length):
        frame = audio_int16[i:i+frame_length]
        if len(frame) == frame_length and vad.is_speech(frame.tobytes(), TARGET_SR):
            voiced.extend(frame)

    voiced = np.array(voiced, dtype=np.int16) / 32768.0
    os.makedirs(os.path.dirname(output_wav), exist_ok=True)
    sf.write(output_wav, voiced, TARGET_SR)

if __name__ == "__main__":
    print("Audio preprocessing with VAD ready.")
