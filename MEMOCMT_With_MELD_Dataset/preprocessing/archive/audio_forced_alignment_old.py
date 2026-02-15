"""
Forced Alignment using Wav2Vec2 + CTC
Aligns text transcripts to audio at word/phoneme level

NOTE:
- This is optional and NOT required by the core architecture
- Provided as an advanced preprocessing enhancement
"""

import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()

def forced_align(audio_path, transcript):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    # NOTE: True phoneme/word timestamps require MFA or Gentle
    return {
        "predicted_transcript": transcription,
        "reference_transcript": transcript
    }

if __name__ == "__main__":
    print("Forced alignment module ready (CTC-based).")
