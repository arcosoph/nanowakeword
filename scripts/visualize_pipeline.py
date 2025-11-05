# reality_gap_analyzer.py

import os
import random
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import torchaudio
from sklearn.metrics.pairwise import cosine_similarity
from nanowakeword.utils.audio_processing import AudioFeatures

def pad_audio(audio_tensor, target_length):
    """
    Padding the audio tensor to a specific length.
    """
    current_length = audio_tensor.shape[1]
    if current_length < target_length:
        padding = target_length - current_length
        padded_audio = torch.nn.functional.pad(audio_tensor, (0, padding))
        return padded_audio
    elif current_length > target_length:
        return audio_tensor[:, :target_length]
    return audio_tensor


def analyze_reality_gap(positive_audio_path, positive_features_path, negative_features_path):
    """
    Analyzes the differences between the real world (ground truth) and the training world.
    """
    print("--- RUNNING REALITY GAP ANALYSIS SCRIPT ---")

    TARGET_LENGTH = 32000 

    try:
       
        print(f"Loading ground truth audio: {os.path.basename(positive_audio_path)}")
        raw_audio_float, sr = torchaudio.load(positive_audio_path)
        if raw_audio_float.shape[0] > 1:
            raw_audio_float = torch.mean(raw_audio_float, dim=0, keepdim=True)

        padded_audio_float = pad_audio(raw_audio_float, TARGET_LENGTH)
        
        raw_audio_int16 = (padded_audio_float * 32767).to(torch.int16)

        print(f"Loading positive training features from: {positive_features_path}")
        pos_features_mmap = np.load(positive_features_path, mmap_mode='r')
        
        print(f"Loading negative training features from: {negative_features_path}")
        neg_features_mmap = np.load(negative_features_path, mmap_mode='r')

    except FileNotFoundError as e:
        print(f"\nERROR: File not found! {e}")
        print("Please ensure you have run the training process to generate feature files.")
        return

    random_pos_feature = pos_features_mmap[random.randint(0, len(pos_features_mmap) - 1)]
    random_neg_feature = neg_features_mmap[random.randint(0, len(neg_features_mmap) - 1)]

    feature_extractor = AudioFeatures(device='cpu')
    ground_truth_feature = feature_extractor.embed_clips(raw_audio_int16.numpy())[0]

    gt_flat = ground_truth_feature.flatten().reshape(1, -1)
    pos_flat = random_pos_feature.flatten().reshape(1, -1)
    neg_flat = random_neg_feature.flatten().reshape(1, -1)
    
    sim_gt_vs_pos = cosine_similarity(gt_flat, pos_flat)[0][0]
    sim_gt_vs_neg = cosine_similarity(gt_flat, neg_flat)[0][0]
    sim_pos_vs_neg = cosine_similarity(pos_flat, neg_flat)[0][0]
    
    print("\n--- Feature Similarity Analysis ---")
    print(f"1. Reality vs. Training (Positive):   {sim_gt_vs_pos:.4f}")
    print(f"2. Reality vs. Training (Negative):   {sim_gt_vs_neg:.4f}")
    print(f"3. Training Positive vs. Negative:    {sim_pos_vs_neg:.4f}")
    print("------------------------------------")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Reality Gap Analysis: Ground Truth vs. Training World", fontsize=16)
    
    img1 = librosa.display.specshow(ground_truth_feature.T, sr=sr, ax=axes[0], x_axis='time', y_axis='mel')
    axes[0].set_title(f"Ground Truth Feature\n(Original Audio, Padded, No Augmentation)")
    fig.colorbar(img1, ax=axes[0])
    
    img2 = librosa.display.specshow(random_pos_feature.T, sr=sr, ax=axes[1], x_axis='time', y_axis='mel')
    axes[1].set_title(f"Random Positive Training Feature\n(Similarity to Ground Truth: {sim_gt_vs_pos:.2f})")
    fig.colorbar(img2, ax=axes[1])
    
    img3 = librosa.display.specshow(random_neg_feature.T, sr=sr, ax=axes[2], x_axis='time', y_axis='mel')
    axes[2].set_title(f"Random Negative Training Feature\n(Similarity to Ground Truth: {sim_gt_vs_neg:.2f})")
    fig.colorbar(img3, ax=axes[2])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("reality_gap_analysis.png")
    plt.close()
    print("\n✅ Analysis complete! Saved 'reality_gap_analysis.png'")

def main():

    positive_audio_dir = os.path.join("nanowakeword_data", "positive_wakeword")
    feature_dir = os.path.join("trained_models", "hey_computer_v1", "1_features")
    
    positive_features_path = os.path.join(feature_dir, "positive_features_train.npy")
    negative_features_path = os.path.join(feature_dir, "negative_features_train.npy")

    ground_truth_audio_file = os.path.join(positive_audio_dir, random.choice(os.listdir(positive_audio_dir)))
    
    analyze_reality_gap(ground_truth_audio_file, positive_features_path, negative_features_path)

if __name__ == "__main__":
    main()