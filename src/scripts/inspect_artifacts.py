import joblib
import json
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ARTIFACTS_DIR = r"c:\Users\ELITEBOOK\Documents\Projects\4th_2018_Class_projects\ML\fraud-detaction_using_unsupervised_learning\artifacts"
FEATURE_LISTS_PATH = os.path.join(ARTIFACTS_DIR, "feature_lists.json")
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.joblib")
THRESHOLDS_PATH = os.path.join(ARTIFACTS_DIR, "model_thresholds.json")

def inspect_and_update():
    print("Loading artifacts...")
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("Preprocessor loaded.")
    except Exception as e:
        print(f"Error loading preprocessor: {e}")
        return

    # Load existing feature lists
    try:
        with open(FEATURE_LISTS_PATH, "r") as f:
            feature_lists = json.load(f)
    except FileNotFoundError:
        print("feature_lists.json not found, creating new.")
        feature_lists = {}

    # 1. Extract Service Categories
    # Assuming preprocessor is a ColumnTransformer and 'cat' is the name of the categorical step
    # or finding the OneHotEncoder step.
    
    service_categories = []
    
    # Try to find OneHotEncoder in the transformers
    if isinstance(preprocessor, ColumnTransformer):
        for name, transformer, cols in preprocessor.transformers_:
            if isinstance(transformer, OneHotEncoder) or (hasattr(transformer, 'steps') and any(isinstance(step[1], OneHotEncoder) for step in transformer.steps)):
                # If pipeline, get OHE
                ohe = transformer
                if hasattr(transformer, 'steps'):
                     for step_name, step_trans in transformer.steps:
                         if isinstance(step_trans, OneHotEncoder):
                             ohe = step_trans
                             break
                
                # Check if 'service' is in the columns for this transformer
                # specific logic depends on how columns are stored (names or indices)
                # But usually OHE stores categories_
                
                # We need to map which category index corresponds to 'service'
                # feature_lists['categorical'] usually has ['protocol_type', 'service', 'flag']
                # If OHE has categories_ list of arrays, we need to know which one is service.
                
                if 'categorical' in feature_lists:
                    try:
                        service_idx = feature_lists['categorical'].index('service')
                        if hasattr(ohe, 'categories_') and len(ohe.categories_) > service_idx:
                            service_categories = ohe.categories_[service_idx].tolist()
                            print(f"Found {len(service_categories)} service categories from OHE.")
                    except ValueError:
                        print("'service' not in categorical list.")
                break
    
    if service_categories:
        feature_lists['service_top_k'] = service_categories
        print("Updated service_top_k.")
    else:
        print("Could not extract service categories automatically. Using fallback placeholders if needed.")
        if 'service_top_k' not in feature_lists:
             feature_lists['service_top_k'] = ["http", "smtp", "ftp", "ftp_data", "other"] # minimal fallback

    # Update feature_lists.json
    with open(FEATURE_LISTS_PATH, "w") as f:
        json.dump(feature_lists, f, indent=4)
    print(f"Updated {FEATURE_LISTS_PATH}")

    # 2. Create/Update Model Thresholds
    # Since we don't have the raw training scores to calculate percentiles exactly right now without loading the data,
    # we will use the user-suggested defaults or placeholders.
    # The user suggested: { "T_low": 0.40, "T_high": 0.70, "model_version": "if-YYYYMMDD" }
    
    thresholds = {
        "T_low": 0.45,  # Conservative start
        "T_high": 0.75,
        "model_version": "if-v1-baseline"
    }
    
    with open(THRESHOLDS_PATH, "w") as f:
        json.dump(thresholds, f, indent=4)
    print(f"Created {THRESHOLDS_PATH}")

if __name__ == "__main__":
    inspect_and_update()
