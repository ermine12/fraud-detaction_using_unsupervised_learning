from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import datetime
import os
import sys
import os
# Add project root to sys.path to allow absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.backend.model_utils import load_artifacts, preprocess_record, get_prediction, compute_top_features

app = FastAPI()

# Mount frontend
app.mount("/static", StaticFiles(directory="src/frontend"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("src/frontend/index.html")

# Global State
artifacts = {}

class PredictionRequest(BaseModel):
    # Flexible dict to accept all UI fields
    # or define specific fields. User said "Accept a JSON body with the 12 UI fields".
    # Using dict to be generic and safe against variations.
    data: dict

class FeedbackRequest(BaseModel):
    request_id: str
    feedback_label: str # 'correct' / 'incorrect'
    actual_label: str = None

@app.on_event("startup")
def startup_event():
    try:
        preprocessor, model, feature_lists, thresholds = load_artifacts()
        artifacts["preprocessor"] = preprocessor
        artifacts["model"] = model
        artifacts["feature_lists"] = feature_lists
        artifacts["thresholds"] = thresholds
    except Exception as e:
        import traceback
        with open("startup_error.txt", "w") as f:
            f.write(traceback.format_exc())
        print("Startup failed. See startup_error.txt")
        # Don't raise, let it start so we can see the file.
        # But endpoints will fail.
    
    # Setup Log files
    os.makedirs("src/backend/logs", exist_ok=True)
    if not os.path.exists("src/backend/logs/requests.csv"):
        with open("src/backend/logs/requests.csv", "w") as f:
            f.write("timestamp,score,label,percentile,input_json\n")
    if not os.path.exists("src/backend/logs/feedback.csv"):
        with open("src/backend/logs/feedback.csv", "w") as f:
            f.write("request_id,timestamp,feedback_label,actual_label\n")

@app.post("/predict")
def predict(request: dict):
    # Expect flattened keys: { "duration": 0, ... }
    # User snippet showed direct keys.
    input_data = request # directly the body
    
    # Validate
    # (Simple validation: check if empty)
    if not input_data:
        raise HTTPException(status_code=400, detail="Empty input")

    try:
        # Preprocess
        X_p = preprocess_record(input_data, artifacts["feature_lists"], artifacts["preprocessor"])
        
        # Predict
        score, label = get_prediction(artifacts["model"], X_p, artifacts["thresholds"])
        
        # Percentile (Placeholder)
        percentile = int(score * 100) # Dummy for now, should use training distribution
        percentile = max(0, min(99, percentile))
        
        # Top Features
        top_features = compute_top_features(input_data, X_p, artifacts["model"], artifacts["feature_lists"])
        
        result = {
            "score": round(float(score), 3),
            "percentile": percentile,
            "label": label,
            "top_features": top_features,
            "model_version": artifacts["thresholds"].get("model_version", "unknown"),
            "notes": "Automated prediction"
        }
        
        # Log
        timestamp = datetime.datetime.now().isoformat()
        import json
        json_str = json.dumps(input_data).replace('"', '""')
        log_line = f'{timestamp},{score},{label},{percentile},"{json_str}"\n'
        with open("src/backend/logs/requests.csv", "a") as f:
            f.write(log_line)
            
        return result

    except Exception as e:
        print(f"Error: {e}")
        # raise HTTPException(status_code=500, detail=str(e)) # Detailed for debugging
        # In production, generic error. For now, detailed.
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
def feedback(data: FeedbackRequest):
    timestamp = datetime.datetime.now().isoformat()
    line = f"{data.request_id},{timestamp},{data.feedback_label},{data.actual_label}\n"
    with open("src/backend/logs/feedback.csv", "a") as f:
        f.write(line)
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8081)
