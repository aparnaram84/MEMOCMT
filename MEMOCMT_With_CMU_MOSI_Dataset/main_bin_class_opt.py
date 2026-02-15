import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from modules.data_loader import load_segmented_data
from modules.feature_extractors import MultiModalFeatureExtractor
from modules.model_cmt import CrossModalTransformer

def robust_parse_transcript(base_dir, vid_id):
    """Retrieves actual text for the segment to ensure unique BERT embeddings."""
    try:
        parent_id, seg_num = vid_id.rsplit('_', 1)
        internal_prefix = f"{seg_num}_"
    except ValueError: return "neutral", False
    
    path = os.path.join(base_dir, "Raw", "Transcript", "Segmented", f"{parent_id}.annotprocessed")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith(internal_prefix):
                    return (line.strip().split(' ')[-1] if ' ' in line else line.strip()), True
    return "neutral", False

def plot_binary_visuals(y_true, y_pred, y_probs, attn_list, title):
    """Generates visualizations with explicitly labeled axes."""
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
    plt.title(f"Confusion Matrix: {title}")
    plt.xlabel("Predicted Sentiment Class"); plt.ylabel("True Sentiment Class")
    plt.show()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
    plt.plot([0,1], [0,1], 'k--')
    plt.title(f"ROC Curve: {title}")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(); plt.show()

    # 3. Cross-Modal Attention Bar Chart
    avg_attn = np.mean(np.vstack(attn_list), axis=0)
    plt.figure(figsize=(6,4))
    sns.barplot(x=['Text', 'Audio', 'Visual'], y=avg_attn, hue=['Text', 'Audio', 'Visual'], palette='viridis', legend=False)
    plt.title(f"Modality Attention Importance: {title}")
    plt.xlabel("Input Modality Type"); plt.ylabel("Attention Weight (Relative Importance)")
    plt.show()

def train():
    print("--- Starting Optimized Binary Pipeline with Early Stopping ---")
    DATA_PATH = r"C:\Mahesh\Dissertation\AparnaThesis\git_mosi_memocmt\MOSI_MEMOCMT\data"
    labels_df = pd.read_csv(os.path.join(DATA_PATH, "labels.csv")).set_index('segment_id')
    train_ids, test_ids = load_segmented_data(DATA_PATH)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossModalTransformer().to(device)
    extractor = MultiModalFeatureExtractor()

    # REGULARIZATION: Higher weight decay and weighted loss to prevent overfitting and bias
    # Calculate pos_weight for BCE: (negative_count / positive_count)
    neg_count = (labels_df.loc[train_ids, 'sentiment'] <= 0).sum()
    pos_count = (labels_df.loc[train_ids, 'sentiment'] > 0).sum()
    pos_weight = torch.tensor([neg_count / pos_count]).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-1)
    
    # SCHEDULER: Reduces learning rate when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # EARLY STOPPING SETTINGS
    best_val_loss = float('inf')
    early_stop_patience = 5
    stop_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    num_epochs = 50 

    for epoch in range(num_epochs):
        # --- TRAINING PHASE (Full Data) ---
        model.train()
        tr_losses, tr_y, tr_p = [], [], []
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        for vid_id in train_ids: # Training on FULL dataset
            if vid_id not in labels_df.index: continue
            txt, found = robust_parse_transcript(DATA_PATH, vid_id)
            audio_p = os.path.join(DATA_PATH, "Raw/Audio/WAV_16000/Segmented", f"{vid_id}.wav")
            video_p = os.path.join(DATA_PATH, "Raw/Video/Segmented", f"{vid_id}.mp4")
            if not (os.path.exists(audio_p) and os.path.exists(video_p) and found): continue

            try:
                fused = torch.stack([
                    extractor.get_text_features(txt).to(device),
                    extractor.get_audio_features(audio_p).to(device),
                    extractor.get_visual_features(video_p).to(device)
                ], dim=1)
                
                optimizer.zero_grad()
                logits, _ = model(fused)
                target_bin = torch.tensor([[1.0 if labels_df.loc[vid_id, 'sentiment'] > 0 else 0.0]]).to(device)
                
                loss = criterion(logits, target_bin); loss.backward(); optimizer.step()
                tr_losses.append(loss.item()); tr_y.append(int(target_bin.item()))
                tr_p.append(1 if torch.sigmoid(logits).item() > 0.5 else 0)
            except Exception: continue
        
        # --- VALIDATION PHASE ---
        model.eval()
        va_losses, va_y, va_p = [], [], []
        with torch.no_grad():
            for vid_id in test_ids[:100]:
                txt_v, found_v = robust_parse_transcript(DATA_PATH, vid_id)
                audio_v = os.path.join(DATA_PATH, "Raw/Audio/WAV_16000/Segmented", f"{vid_id}.wav")
                video_v = os.path.join(DATA_PATH, "Raw/Video/Segmented", f"{vid_id}.mp4")
                if not (os.path.exists(audio_v) and os.path.exists(video_v) and found_v): continue

                try:
                    f_v = torch.stack([
                        extractor.get_text_features(txt_v).to(device), 
                        extractor.get_audio_features(audio_v).to(device), 
                        extractor.get_visual_features(video_v).to(device)
                    ], dim=1)
                    v_logits, _ = model(f_v)
                    v_target = torch.tensor([[1.0 if labels_df.loc[vid_id, 'sentiment'] > 0 else 0.0]]).to(device)
                    va_losses.append(criterion(v_logits, v_target).item())
                    va_y.append(int(v_target.item())); va_p.append(1 if torch.sigmoid(v_logits).item() > 0.5 else 0)
                except Exception: continue
        
        # LOGGING AND SCHEDULING
        avg_va_loss = np.mean(va_losses)
        print(f"Train Loss: {np.mean(tr_losses):.4f} | Acc: {accuracy_score(tr_y, tr_p)*100:.2f}%")
        print(f"Val Loss: {avg_va_loss:.4f} | Acc: {accuracy_score(va_y, va_p)*100:.2f}% | F1: {f1_score(va_y, va_p):.4f}")
        
        scheduler.step(avg_va_loss)

        # EARLY STOPPING CHECK
        if avg_va_loss < best_val_loss:
            best_val_loss = avg_va_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            stop_counter = 0
            print("--> Validation Improved. Model weight updated.")
        else:
            stop_counter += 1
            if stop_counter >= early_stop_patience:
                print(f"EARLY STOPPING after {epoch+1} epochs.")
                break

    # --- FINAL TEST EVALUATION ---
    model.load_state_dict(best_model_wts)
    model.eval()
    te_y, te_p, te_prob, te_attn = [], [], [], []
    print("\n--- FINAL TEST LOGS (Best Model) ---")
    with torch.no_grad():
        for vid_id in test_ids[100:130]:
            try:
                txt_t, _ = robust_parse_transcript(DATA_PATH, vid_id)
                # ... extraction logic ...
                logits, attn = model(f_t)
                prob = torch.sigmoid(logits).item()
                actual_score = labels_df.loc[vid_id, 'sentiment']
                print(f"ID: {vid_id} | True: {actual_score:+.2f} | Pred: {logits.item():+.2f} | Class: {1 if prob > 0.5 else 0}")
                
                te_prob.append(prob); te_y.append(1 if actual_score > 0 else 0)
                te_p.append(1 if prob > 0.5 else 0); te_attn.append(attn.cpu().numpy())
            except Exception: continue

    if te_y: plot_binary_visuals(te_y, te_p, te_prob, te_attn, "Optimized Model")

if __name__ == "__main__":
    train()