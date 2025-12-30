import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import IsolationForest

# Config
DATA_PATH = r"c:\Users\ELITEBOOK\Documents\Projects\4th_2018_Class_projects\ML\fraud-detaction_using_unsupervised_learning\data\raw\KDDTrain+.txt"
ARTIFACTS_DIR = r"c:\Users\ELITEBOOK\Documents\Projects\4th_2018_Class_projects\ML\fraud-detaction_using_unsupervised_learning\artifacts"
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.joblib")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "isolation_forest.joblib")
FEATURE_LISTS_PATH = os.path.join(ARTIFACTS_DIR, "feature_lists.json")

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

def retrain():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, names=COLUMNS)
    
    # Filter only normals? User said "training only on normal traffic" in conversation history.
    # Usually Unsupervised Anomaly Detection is trained on normal data (semi-supervised) or mixed.
    # Isolation Forest works on mixed, but better on normal.
    # The prompt says: "preprocessing & models exist...". I should replicate what was likely done.
    # I'll train on ALL data or just normals?
    # IF assumes contamination.
    # Let's train on Normals only to be safe/clean for anomaly detection.
    # Or follow "artifacts/isolation_forest.joblib" config if known.
    # I'll just train on normals to build a "profile of normality".
    
    df_normal = df[df['label'] == 'normal']
    print(f"Training on {len(df_normal)} normal records.")

    # Feature Selection
    # Use keys from feature_lists.json `selected_for_modeling` if available.
    with open(FEATURE_LISTS_PATH, "r") as f:
        feats = json.load(f)
    
    selected_cols = feats.get("selected_for_modeling", [])
    numeric_cols = feats.get("numeric_cols", []) # or 'numeric'
    if not numeric_cols: numeric_cols = feats.get("numeric", [])
    categorical_cols = feats.get("categorical_cols", [])
    if not categorical_cols: categorical_cols = feats.get("categorical", [])
    
    # Intersection with selected
    final_numeric = [c for c in selected_cols if c in numeric_cols]
    final_categorical = [c for c in selected_cols if c in categorical_cols]
    
    if not final_numeric and not final_categorical:
        # Fallback if selected keys missing
        final_numeric = ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count', 'serror_rate', 'dst_host_count']
        final_categorical = ['protocol_type', 'service', 'flag']
        print("Using fallback features.")
    
    # Pipeline
    # 1. Log Transform for specific skewed cols? (src_bytes, dst_bytes)
    # Keeping it simple: Standard Scaler for numeric, OneHot for categorical.
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse=False for IF compatibility sometimes easier
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, final_numeric),
            ('cat', categorical_transformer, final_categorical)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    
    model = IsolationForest(
        n_estimators=100, 
        contamination=0.01, 
        random_state=42, 
        n_jobs=-1
    )
    
    print("Fitting preprocessor...")
    X = df_normal[final_numeric + final_categorical]
    X_processed = preprocessor.fit_transform(X)
    
    print("Fitting model...")
    model.fit(X_processed)
    
    print("Saving artifacts...")
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(model, MODEL_PATH)
    
    # Update feature_lists with exact service categories found in training
    # (Though we already updated it from test, training is authoritative)
    # Let's trust existing feature_lists.json for now.
    
    print("Retraining complete.")

if __name__ == "__main__":
    retrain()
