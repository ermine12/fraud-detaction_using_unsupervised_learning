import joblib
import os
import traceback

ARTIFACTS_DIR = r"c:\Users\ELITEBOOK\Documents\Projects\4th_2018_Class_projects\ML\fraud-detaction_using_unsupervised_learning\artifacts"
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.joblib")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "isolation_forest.joblib")

def debug_load():
    print("Testing Preprocessor Load...")
    try:
        p = joblib.load(PREPROCESSOR_PATH)
        print("Preprocessor loaded successfully.")
    except Exception:
        traceback.print_exc()

    print("\nTesting Model Load...")
    try:
        m = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    debug_load()
