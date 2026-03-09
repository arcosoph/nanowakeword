import os
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from tabulate import tabulate

# ==============================
# CONFIG
# ==============================

MODEL_PATHS = [
    "trained_models/arcosoph_A_v1/model/arcosoph_A_v1.onnx",
]

POSITIVE_NPY = "./trained_models/arcosoph_A_v1/features/positive_features.npy"
NEGATIVE_NPY = "./trained_models/arcosoph_A_v1/features/negative_features.npy"

THRESHOLD = 0.95
MAX_SAMPLES = None
BATCH_SIZE = 128

# ==============================

def load_serial(path, limit=None):
    data = np.load(path)
    if limit:
        data = data[:limit]
    return data.astype(np.float32)


from tqdm import tqdm

def evaluate_model(session, pos_data, neg_data):

    input_name = session.get_inputs()[0].name

    # -------- Positive --------
    misses = 0
    total_pos_batches = (len(pos_data) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(range(0, len(pos_data), BATCH_SIZE),
                  total=total_pos_batches,
                  desc="Positive",
                  leave=False):

        batch = pos_data[i:i+BATCH_SIZE]
        outputs = session.run(None, {input_name: batch})[0]
        scores = outputs.squeeze()
        misses += np.sum(scores < THRESHOLD)

    # -------- Negative --------
    false_alarms = 0
    total_neg_batches = (len(neg_data) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(range(0, len(neg_data), BATCH_SIZE),
                  total=total_neg_batches,
                  desc="Negative",
                  leave=False):

        batch = neg_data[i:i+BATCH_SIZE]
        outputs = session.run(None, {input_name: batch})[0]
        # print("Batch shape:", batch.shape)
        # print("Model expects:", session.get_inputs()[0].shape)
        scores = outputs.squeeze()
        false_alarms += np.sum(scores > THRESHOLD)

    return misses, false_alarms

# ==============================
# MAIN
# ==============================

pos_data = load_serial(POSITIVE_NPY, MAX_SAMPLES)
neg_data = load_serial(NEGATIVE_NPY, MAX_SAMPLES)

results = []

for model_path in MODEL_PATHS:

    if not os.path.exists(model_path):
        continue

    print(f"Evaluating: {os.path.basename(model_path)}")

    session = ort.InferenceSession(model_path)

    misses, false_alarms = evaluate_model(session, pos_data, neg_data)

    total_error = misses + false_alarms

    results.append([
        os.path.basename(model_path),
        len(pos_data),
        misses,
        len(neg_data),
        false_alarms,
        total_error
    ])

# Ranking
results.sort(key=lambda x: x[-1])

print("\nFINAL RANKING\n")
print(tabulate(results, headers=[
    "Model",
    "Pos Total",
    "Miss",
    "Neg Total",
    "False Alarm",
    "Total Error"
], tablefmt="pretty"))