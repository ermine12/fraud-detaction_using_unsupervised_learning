# Unsupervised NIDS Project

An anomaly detection system for network intrusion using Isolation Forest. This project uses a FastAPI backend (unsupervised learner) to flag suspicious network traffic patterns.

## 🚀 How to Run

### Prerequisites
- Python 3.9+
- Dependencies: `pip install -r requirements.txt`

### Start the Application
Run the backend server (which also serves the frontend):
```bash
uvicorn src.backend.main:app --reload
```
or
```bash
python src/backend/main.py
```
Access the UI at: **http://127.0.0.1:8000/**

## 🧪 Testing
A set of confirmed test cases (normal and attack examples) is available in `data/processed/ui_test_cases.json`.
1. Open the UI.
2. Manually input values from a test case.
3. Click "Predict Status".
4. Verify the result matches the expected label (Normal vs Anomaly).

## 📂 Project Structure
- `src/backend/`: Server logic (`main.py`) and model utilities.
- `src/frontend/`: Static Web UI (`index.html`).
- `artifacts/`: Trained models (`isolation_forest.joblib`) and feature lists.
- `logs/`: Request logs and user feedback (saved in `src/backend/logs/`).

## ⚠️ Disclaimer
**This is a statistical anomaly detector — not a definitive attack verdict.**
High scores indicate unusual traffic patterns compared to the training baseline (normal traffic). Always verify with domain expertise.
