import numpy as np
import matplotlib.pyplot as plt
import os
import random

# file path
CURRENT_NPY_PATH = r"trained_models\arcosoph\features\positive_features_train.npy"

def plot_features(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    print(f"📂 Loading: {os.path.basename(file_path)}...")
    
    try:
        # Load the data
        data = np.load(file_path, mmap_mode='r')
        total_samples = data.shape[0]
        
        print(f"   Shape: {data.shape} (Samples, Time, Channels)")
        
        # Randomly select 3 samples to visualize
        indices = random.sample(range(total_samples), 3)
        
        plt.figure(figsize=(15, 10))
        
        for i, idx in enumerate(indices):
            # Get one audio sample
            sample = data[idx] 
            
            # Transpose for plotting (Frequency usually on Y-axis, Time on X-axis)
            # Our shape is (Time, Freq), so .T makes it (Freq, Time)
            sample_to_plot = sample.T 
            
            plt.subplot(3, 1, i + 1)
            
            # Plot Heatmap
            # origin='lower' puts low frequencies at the bottom (Standard for Spectrograms)
            im = plt.imshow(sample_to_plot, aspect='auto', origin='lower', cmap='viridis')
            
            plt.colorbar(im, label='Intensity (dB)')
            plt.title(f"Sample Index: {idx} (Shape: {sample.shape})")
            plt.ylabel("Frequency / Channels")
            plt.xlabel("Time Frames")
            
        plt.tight_layout()
        plt.show()
        
        print("✅ Visualization complete. Check the popup window.")

    except Exception as e:
        print(f"❌ Error reading file: {e}")

if __name__ == "__main__":
    plot_features(CURRENT_NPY_PATH)