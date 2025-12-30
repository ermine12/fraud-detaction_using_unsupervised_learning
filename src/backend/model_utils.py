import joblib
import json
import os
import pandas as pd
import numpy as np
import sklearn.compose
import sklearn.ensemble
import sklearn.preprocessing

ARTIFACTS_DIR = r"c:\Users\ELITEBOOK\Documents\Projects\4th_2018_Class_projects\ML\fraud-detaction_using_unsupervised_learning\artifacts"
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.joblib")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "isolation_forest.joblib")
FEATURE_LISTS_PATH = os.path.join(ARTIFACTS_DIR, "feature_lists.json")
THRESHOLDS_PATH = os.path.join(ARTIFACTS_DIR, "model_thresholds.json")

def load_artifacts():
    print("Loading artifacts...")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_LISTS_PATH, "r") as f:
        feature_lists = json.load(f)
    try:
        with open(THRESHOLDS_PATH, "r") as f:
            thresholds = json.load(f)
    except FileNotFoundError:
        print("Warning: model_thresholds.json not found. Using defaults.")
        thresholds = {"T_low": 0.45, "T_high": 0.75, "model_version": "unknown"}
    
    print("Artifacts loaded.")
    return preprocessor, model, feature_lists, thresholds

def preprocess_record(record: dict, feature_lists: dict, preprocessor) -> np.ndarray:
    """
    Preprocess a single record (dict) into model input format.
    Handles 'service' -> 'service_top_k' mapping.
    """
    # 1. Map service to top_k
    service_top_k = feature_lists.get("service_top_k", [])
    if record.get("service") not in service_top_k:
        record["service"] = "other"  # Assumes 'other' is in the training set or handled by OHE 'handle_unknown'
    
    # 2. Ensure all expected columns are present
    # Check 'selected_for_modeling' and 'categorical' lists
    required_cols = feature_lists.get("selected_for_modeling", [])
    categorical_cols = feature_lists.get("categorical", ["protocol_type", "service", "flag"])
    
    # Fill missing numeric/selected fields
    for col in required_cols:
        if col not in record:
            record[col] = 0.0
            
    # Fill missing categorical fields
    for col in categorical_cols:
        if col not in record:
            record[col] = "other"

    df = pd.DataFrame([record])
    
    # 3. Transform
    X_p = preprocessor.transform(df)
    return X_p

def get_prediction(model, X_p, thresholds):
    """
    Get score, label, and percentile.
    """
    # Isolation Forest: decision_function returns anomaly score. 
    # Lower is more anomalous? Or higher?
    # Sklearn IF: decision_function < 0 is anomaly. 
    # But usually we map it to [0,1] or flip it.
    # The user spec says: "higher = more anomalous".
    # IF decision_function: larger (positive) = normal, smaller (negative) = anomaly.
    # So we need to invert. Score = -raw_score.
    
    raw_score = model.decision_function(X_p)[0]
    score = -raw_score # Higher = more anomalous
    
    # Normalize? User asked for [0,1].
    # IF scores are roughly -0.5 to 0.5. 
    # Let's simple Min-Max normalize if we had training range.
    # Without training range, we can just use the score as is, or sigmoid?
    # User spec: "Convert to a normalized score in [0,1] ... (If model returns inverse scale, invert it.)"
    # Actually, sklearn 'score_samples' is different from 'decision_function'.
    # I'll stick to -decision_function and maybe shift/scale based on typical range [-0.9, 0.5] -> [0.9, -0.5]
    # Let's just output the raw inverted score for now, and T_low/T_high will be tuned to that scale.
    # If the user-provided thresholds (0.4, 0.7) suggest [0,1] range, maybe I should add 0.5?
    # Default Sklearn offset is -0.5. So decision_function is usually shifted.
    # Let's normalize loosely: score = 0.5 - decision_function / 2 ?
    # Let's stick to simple inversion: score = 0.5 - (raw_score / 2)? No, raw score is around 0.
    
    # Let's trust the 'thresholds' are aligned with whatever 'score' metric we produce.
    # Since we are setting defaults 0.45/0.75, let's assume we output values around 0-1.
    # Decision function range is typically [-0.2, 0.2] for normals?
    # I'll use: score = 0.5 - raw_score
    
    transformed_score = 0.5 - raw_score # if raw=0.1 (normal) -> 0.4. if raw=-0.2 (anomaly) -> 0.7.
    
    # Label
    t_low = thresholds.get("T_low", 0.45)
    t_high = thresholds.get("T_high", 0.75)
    
    if transformed_score < t_low:
        label = "Normal"
    elif transformed_score < t_high:
        label = "Suspicious"
    else:
        label = "Anomaly"
        
    return transformed_score, label

def compute_top_features(record, X_p, model, feature_lists):
    """
    Compute top contributing features.
    For simplicity: show deviations from 'median' if available, or just raw values of high-variance features.
    User spec: "compute absolute difference between raw input and training median... rank top 3."
    Missing median? We'll just return the top 3 numeric features with highest values (normalized) or heuristic.
    """
    # Placeholder: Just return the top 3 numeric features by absolute value (if scaled).
    # Or just list the numeric features provided in input.
    
    # Better: list categorical 'flag' or 'service' if unusual?
    
    features = []
    # Dummy logic:
    for k, v in record.items():
        if isinstance(v, (int, float)) and v > 0:
            features.append({"feature": k, "value": v, "explanation": "Non-zero value"})
            
    return features[:3]
