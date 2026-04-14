import os
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from tabulate import tabulate
import warnings

# ==============================
# CONFIG
# ==============================
# MODEL_FOLDER = "trained_models"
# MODEL_PATHS = glob.glob(os.path.join(MODEL_FOLDER, "**/*.onnx"), recursive=True)
# MODEL_FOLDER = "trained_models"
MODEL_PATHS = [
    './trained_models/tadano_A_v192/model/tadano_A_v192.onnx',
    './trained_models/tadano_A_v193/model/tadano_A_v193.onnx',
    './trained_models/tadano_A_v194/model/tadano_A_v194.onnx',
    './trained_models/tadano_A_v195/model/tadano_A_v195.onnx',

]

POSITIVE_NPY = './trained_models/tadano_A_v176/features/positive_features_train.npy'
NEGATIVE_NPY = "RACON_11h_v1.npy"

THRESHOLD = 0.90
MAX_SAMPLES = None
BATCH_SIZE = 328

# ==============================

def load_serial(path, limit=None):
    """Numpy loads the file and converts it to float32."""
    try:
        data = np.load(path)
        if limit:
            data = data[:limit]
        return data.astype(np.float32)
    except FileNotFoundError:
        print(f"[X] Error: File not found - {path}")
        return None

def evaluate_model(session, pos_data, neg_data, batch_size, model_name):
    """
    Evaluates the model using a specified batch size.
    Will throw an Exception if there is a problem.
    """
    input_name = session.get_inputs()[0].name

    # Positive Data Evaluation 
    misses = 0
    total_pos_batches = (len(pos_data) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(pos_data), batch_size),
                  total=total_pos_batches,
                  desc=f"Positive (BS={batch_size})",
                  leave=False):
        batch = pos_data[i:i + batch_size]
        outputs = session.run(None, {input_name: batch})[0]
        scores = outputs.squeeze()
        misses += np.sum(scores < THRESHOLD)

    # Negative Data Evaluation 
    false_alarms = 0
    total_neg_batches = (len(neg_data) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(neg_data), batch_size),
                  total=total_neg_batches,
                  desc=f"Negative (BS={batch_size})",
                  leave=False):
        batch = neg_data[i:i + batch_size]
        outputs = session.run(None, {input_name: batch})[0]
        scores = outputs.squeeze()
        false_alarms += np.sum(scores > THRESHOLD)

    return misses, false_alarms

# ==============================
# MAIN
# ==============================

print("Loading data...")
pos_data = load_serial(POSITIVE_NPY, MAX_SAMPLES)
neg_data = load_serial(NEGATIVE_NPY, MAX_SAMPLES)

if pos_data is None or neg_data is None:
    print("Unable to load required data file. Closing program...")
    exit()

results = []

for model_path in MODEL_PATHS:
    model_name = os.path.basename(model_path)
    print(f"\nEvaluating: {model_name}")

    if not os.path.exists(model_path):
        warnings.warn(f"Model file '{model_path}' Not found. Skipping...")
        continue

    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        try:
            misses, false_alarms = evaluate_model(session, pos_data, neg_data, BATCH_SIZE, model_name)
        
        except Exception as e:
            print(f"⚠️ Warning: '{model_name}' failed with default batch size ({BATCH_SIZE}). Error: {e}")
            print("[*] Retrying with batch size 1...")

            try:
                misses, false_alarms = evaluate_model(session, pos_data, neg_data, 1, model_name)
            
            except Exception as e2:
                print(f"❌ Failed: '{model_name}' could not be evaluated even with batch size 1. Error: {e2}")
                results.append([
                    model_name,
                    len(pos_data), "FAILED",
                    len(neg_data), "FAILED",
                    "N/A"
                ])
                continue

        total_error = misses + false_alarms
        results.append([
            model_name,
            len(pos_data), misses,
            len(neg_data), false_alarms,
            total_error
        ])

    except Exception as e:
        print(f"The model '{model_name}' could not be loaded. Error: {e}")
        results.append([
            model_name,
            "N/A", "LOAD FAILED",
            "N/A", "LOAD FAILED",
            "N/A"
        ])

results.sort(key=lambda x: (isinstance(x[-1], str), x[-1]))

print(f"\n\n{'='*15} Final Ranking {'='*15}\n")
print(tabulate(results, headers=[
    "Model",
    "Pos Total",
    "Miss",
    "Neg Total",
    "False Alarm",
    "Total Error"
], tablefmt="pretty"))