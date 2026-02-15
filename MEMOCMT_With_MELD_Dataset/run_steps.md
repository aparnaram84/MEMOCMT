✅ End-to-End Execution Pipeline

AVT-Improved MemoCMT on MELD

STEP 0️⃣ Environment Setup (Once)
0.1 Create & activate environment
conda create -n avt_memocmt python=3.9
conda activate avt_memocmt

0.2 Install dependencies
pip install -r requirements.txt


Verify:

python -c "import torch; print(torch.__version__)"

STEP 1️⃣ Dataset Setup (MELD)
1.1 Download MELD

From:

https://affective-meld.github.io/

1.2 Expected directory structure
data/MELD/
├── train/
│   ├── audio/
│   ├── video/
│   └── text/
├── dev/
└── test/


Each utterance should have:

.wav (audio)

.mp4 (video)

.txt or CSV entry (text + label)

STEP 2️⃣ Preprocessing Pipeline

⚠️ Preprocessing improves signal quality
✔ Does NOT change architecture

2.1 Audio preprocessing (basic)
python preprocessing/audio_preprocess.py

2.2 Audio preprocessing with VAD (recommended)
python preprocessing/audio_preprocess_vad.py


Output:

data/MELD/train/audio_clean/

2.3 (Optional) Forced alignment
python preprocessing/audio_forced_alignment.py


Use only for analysis / XAI, not required for training.

2.4 Video preprocessing (basic frames)
python preprocessing/video_preprocess.py

2.5 Video preprocessing with face detection (recommended)
python preprocessing/video_preprocess_mtcnn.py


Output:

data/MELD/train/video_faces/

2.6 Text preprocessing
python preprocessing/text_preprocess.py

STEP 3️⃣ Feature Extraction

Converts raw signals → embeddings
✔ Mandatory for Cross-Modal Transformer

3.1 Audio features (HuBERT)
python features/extract_audio_features.py


Output:

features/audio_features.pt

3.2 Visual features (ResNet)
python features/extract_visual_features.py


Output:

features/visual_features.pt

3.3 Text features (BERT)
python features/extract_text_features.py


Output:

features/text_features.pt

STEP 4️⃣ Model Training

Executes full AVT architecture

4.1 Configure hyperparameters

Edit:

config.yaml


Key fields:

training:
  batch_size: 8
  num_epochs: 20
  learning_rate: 2e-5

4.2 Start training
python train.py


What happens internally:

Load audio, visual, text embeddings

Project to 256-D shared space

Cross-Modal Transformer (A↔V↔T)

Temporal Transformer

Backpropagation

Saved model:

experiments/checkpoints/avt_memocmt.pt

STEP 5️⃣ Evaluation
python evaluate.py


Outputs:

Accuracy

F1-score

UAR

Saved to:

experiments/results.csv
experiments/confusion_matrix.png

STEP 6️⃣ Explainability (XAI)
6.1 Attention maps
python explainability/attention_visualization.py


Saved to:

experiments/attention_maps/

6.2 Grad-CAM (Visual)
python explainability/gradcam.py


Output:

Face heatmaps highlighting emotion-relevant regions

6.3 Integrated Gradients (Text)
python explainability/integrated_gradients.py


Output:

Token-level importance scores

STEP 7️⃣ Inference (Real-Time / Offline)
python inference.py


Pipeline:

Preprocess input

Extract features

Run AVT MemoCMT

Output emotion class

Latency: ~44 ms (GPU)

STEP 8️⃣ Results & Documentation

You now have:

experiments/
├── results.csv
├── confusion_matrix.png
└── attention_maps/


===============================

fix blow issue 
"(avt_memocmt) C:\Mahesh\Dissertation\AparnaThesis\AVT-Improved-MemoCMT-MELD>python preprocessing/audio_preprocess.py

❌ ERROR: ffmpeg not found in PATH
👉 Install ffmpeg and add it to PATH
👉 Windows guide: https://www.gyan.dev/ffmpeg/builds/"

This error is expected and correct behavior after we added the safety check.
It means ffmpeg is not installed or not visible to Python on Windows.

Below is the exact, guaranteed fix for Windows. Follow it once and this issue will never reappear.

✅ Root Cause (Confirmed)
❌ ERROR: ffmpeg not found in PATH


✔ Your script is correct
❌ Windows cannot find ffmpeg.exe

Python uses:

shutil.which("ffmpeg")


This returns None → ffmpeg is not in PATH.

✅ FIX (Windows – Step by Step, No Guessing)
STEP 1️⃣ Download ffmpeg (Correct Source)

Download static Windows build from:

👉 https://www.gyan.dev/ffmpeg/builds/

Choose:

ffmpeg-git-full.7z

STEP 2️⃣ Extract ffmpeg

Extract the archive to:

C:\ffmpeg\


You must see:

C:\ffmpeg\bin\ffmpeg.exe


⚠️ If ffmpeg.exe is not inside bin, the setup is wrong.

STEP 3️⃣ Add ffmpeg to Windows PATH (CRITICAL)

Press Win + R → type:

sysdm.cpl


Go to Advanced → Environment Variables

Under System variables, select Path

Click Edit → New

Add exactly:

C:\ffmpeg\bin


Click OK → OK → OK

Close ALL terminals and VS Code

Reopen terminal

STEP 4️⃣ Verify (DO NOT SKIP)

In a new terminal, run:

ffmpeg -version

Expected output:
ffmpeg version 6.x.x ...


If this works → ✅ ffmpeg is correctly installed.

STEP 5️⃣ Re-run your script
python preprocessing/audio_preprocess.py

Expected output:
[train] Found XXX videos
✅ Audio extracted for train
[dev] Found XXX videos
✅ Audio extracted for dev
[test] Found XXX videos
✅ Audio extracted for test

📂 Confirm Output (Very Important)

You should now see:

data/MELD/train/audio/*.wav
data/MELD/dev/audio/*.wav
data/MELD/test/audio/*.wav


These .wav files are mandatory for:

HuBERT

wav2vec

Audio feature extraction

🧠 Why this is unavoidable (for viva/review)

“MELD stores audio inside video files. Therefore, ffmpeg is required to extract the audio stream. This is a dataset constraint, not a modeling choice.”

⚠️ Alternative (NOT recommended, but possible)

You could hardcode the path:

FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"