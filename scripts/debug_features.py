# debug_features.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import os

print("--- RUNNING FEATURE ANALYSIS SCRIPT ---")

FEATURE_DIR = "./trained_models/hey_computer_v1" #<-- your model path
POSITIVE_FEATURES_PATH = os.path.join(FEATURE_DIR, "positive_features_train.npy")
NEGATIVE_FEATURES_PATH = os.path.join(FEATURE_DIR, "negative_features_train.npy")
# -----------------------------------------

def analyze_features(positive_path, negative_path):
    try:
        print(f"Loading positive features from: {positive_path}")
        positive_features = np.load(positive_path, mmap_mode='r')
        print(f"Loading negative features from: {negative_path}")
        negative_features = np.load(negative_path, mmap_mode='r')
    except FileNotFoundError as e:
        print(f"\nERROR: File not found! {e}")
        print("Please ensure you have run the feature generation step first.")
        return

    print(f"\nPositive features shape: {positive_features.shape}")
    print(f"Negative features shape: {negative_features.shape}")

    random_pos_idx = np.random.randint(0, len(positive_features))
    random_neg_idx = np.random.randint(0, len(negative_features))

    pos_sample = positive_features[random_pos_idx]
    neg_sample = negative_features[random_neg_idx]

    print("\nVisualizing one random positive vs. negative feature map...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = ax1.imshow(pos_sample.T, aspect='auto', origin='lower')
    ax1.set_title("RANDOM POSITIVE FEATURE")
    fig.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(neg_sample.T, aspect='auto', origin='lower')
    ax2.set_title("RANDOM NEGATIVE FEATURE")
    fig.colorbar(im2, ax=ax2)
    
    plt.savefig("debug_feature_comparison.png")
    plt.close()
    print("Saved 'debug_feature_comparison.png'. Do they look visually different?")

    pos_flat = pos_sample.flatten().reshape(1, -1)
    neg_flat = neg_sample.flatten().reshape(1, -1)
    
    similarity = cosine_similarity(pos_flat, neg_flat)[0][0]
    
    print(f"\nCosine Similarity between the two samples: {similarity:.4f}")

if __name__ == "__main__":
    analyze_features(POSITIVE_FEATURES_PATH, NEGATIVE_FEATURES_PATH)