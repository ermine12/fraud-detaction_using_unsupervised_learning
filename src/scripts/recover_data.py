import pandas as pd
import json
import joblib
import os
import numpy as np

# NSL-KDD Column Names (Standard)
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", 
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", 
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", 
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", 
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", 
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", 
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", 
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", 
    "label", "score"
]

ARTIFACTS_DIR = r"c:\Users\ELITEBOOK\Documents\Projects\4th_2018_Class_projects\ML\fraud-detaction_using_unsupervised_learning\artifacts"
DATA_PATH = r"c:\Users\ELITEBOOK\Documents\Projects\4th_2018_Class_projects\ML\fraud-detaction_using_unsupervised_learning\data\raw\KDDTest+.txt"
FEATURE_LISTS_PATH = os.path.join(ARTIFACTS_DIR, "feature_lists.json")
THRESHOLDS_PATH = os.path.join(ARTIFACTS_DIR, "model_thresholds.json")
TEST_CASES_PATH = r"c:\Users\ELITEBOOK\Documents\Projects\4th_2018_Class_projects\ML\fraud-detaction_using_unsupervised_learning\data\processed\ui_test_cases.json"

def recover():
    print(f"Reading {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH, names=COLUMNS)
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    # 1. Update Service List
    print("Extracting services...")
    services = df['service'].value_counts().head(20).index.tolist()
    # Ensure 'other' is handled in backend logic, here just list top k
    
    with open(FEATURE_LISTS_PATH, "r") as f:
        feats = json.load(f)
    
    feats['service_top_k'] = services
    # Ensure numeric_cols is present (if missing, use selected_for_modeling intsersect numeric)
    # The existing file had 'numeric' key, user asked for 'numeric_cols'. I should add alias.
    if 'numeric' in feats:
        feats['numeric_cols'] = feats['numeric']
    if 'categorical' in feats:
        feats['categorical_cols'] = feats['categorical']

    with open(FEATURE_LISTS_PATH, "w") as f:
        json.dump(feats, f, indent=4)
    print("Updated feature_lists.json")

    # 2. Create UI Test Cases
    # 3 Normal, 2 Attack
    # We need to filter by label. In KDDTest+, label is usually 'normal' or attack type.
    
    normals = df[df['label'] == 'normal'].head(3)
    attacks = df[df['label'] != 'normal'].head(2)
    
    test_cases_df = pd.concat([normals, attacks])
    
    # Selecting columns expected by UI? 
    # User said "12 UI fields". Let's try to match them.
    # If not sure, output ALL fields, the backend validation will tell us or we can update backend.
    # Ideally, we include the columns present in 'selected_for_modeling' + categoricals.
    
    selected_cols = feats.get('selected_for_modeling', [])
    cats = feats.get('categorical', [])
    ui_cols = list(set(selected_cols + cats))
    
    # Filter columns if they exist in df
    final_cols = [c for c in ui_cols if c in df.columns]
    
    # Convert to list of dicts
    test_cases = test_cases_df[final_cols].to_dict(orient='records')
    
    # Add labels for reference in the test case file (not for input to predict)
    for i, row in enumerate(test_cases):
        row['_expected_label'] = test_cases_df.iloc[i]['label']
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(TEST_CASES_PATH), exist_ok=True)
    
    with open(TEST_CASES_PATH, "w") as f:
        json.dump(test_cases, f, indent=4)
    print("Created ui_test_cases.json")

    # 3. Create Thresholds (Default)
    defaults = { "T_low": 0.45, "T_high": 0.75, "model_version": "if-v1-recovery" }
    with open(THRESHOLDS_PATH, "w") as f:
        json.dump(defaults, f, indent=4)
    print("Created model_thresholds.json")

if __name__ == "__main__":
    recover()
