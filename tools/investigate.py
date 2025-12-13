
# investigate.py
# -----------------------------------------
# Description:
# This script investigates the environment for your project.
# It checks:
# 1. The location of this script file.
# 2. The current working directory of the terminal.
# 3. Contents of the script directory.
# 4. Contents of the 'trained_models' folder.
# 5. Contents of a specific model folder 'covas_v0.1'.
# 6. Finally, it verifies whether the expected model file exists.
# -----------------------------------------

import os

print("--- Starting Investigation ---")

# 1. Where is this script file located?
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
print(f"\n1. Script Location (SCRIPT_DIR):\n   '{script_dir}'")

# 2. What is the terminal's current working directory?
current_working_dir = os.getcwd()
print(f"\n2. Current Working Directory (os.getcwd()):\n   '{current_working_dir}'")

# 3. What is inside the root folder (SCRIPT_DIR)?
print("\n3. Contents of the Root Folder (SCRIPT_DIR):")
try:
    for item in os.listdir(script_dir):
        print(f"   - {item}")
except Exception as e:
    print(f"   ❌ Could not read directory. Error: {e}")

# 4. What is inside the 'trained_models' folder?
trained_models_dir = os.path.join(script_dir, "trained_models")
print(f"\n4. Contents of '{trained_models_dir}':")
try:
    for item in os.listdir(trained_models_dir):
        print(f"   - {item}")
except Exception as e:
    print(f"   ❌ Could not read directory. Error: {e}")

# 5. What is inside the 'covas_v0.1' folder?
covas_dir = os.path.join(trained_models_dir, "covas_v0.1")
print(f"\n5. Contents of '{covas_dir}':")
try:
    for item in os.listdir(covas_dir):
        print(f"   - {item}")
except Exception as e:
    print(f"   ❌ Could not read directory. Error: {e}")

# 6. Final check: Can the model file actually be found?
model_path = os.path.join(covas_dir, "covas_v0.1.onnx")
print(f"\n6. Final Check for Model File:\n   '{model_path}'")
if os.path.exists(model_path):
    print("\n✅✅✅ SUCCESS: The model file was found by the script!")
else:
    print("\n❌❌❌ FAILURE: The script could NOT find the model file at the specified path.")

print("\n--- Investigation Complete ---")
