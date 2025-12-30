PROJECT REPORT: UNSUPERVISED NETWORK INTRUSION DETECTION SYSTEM

INTRODUCTION
This report documents the end-to-end development of an Unsupervised Network Intrusion Detection System (NIDS) using the NSL-KDD dataset. The goal was to build a system capable of flagging anomalous network traffic patterns without relying on labeled attack data during training, simulating a real-world zero-day threat detection scenario.

PHASE 1: PROJECT SETUP AND EXPLORATORY DATA ANALYSIS (EDA)
We began by initializing the project structure, creating directories for data (raw/processed), notebooks, source code, artifacts, and reports. 

An extensive Exploratory Data Analysis (EDA) was conducted in 'notebooks/EDA_NSL_KDD.ipynb'. Key findings included:
- The dataset contains no missing values but has highly skewed distributions in features like 'src_bytes' and 'dst_bytes'.
- Categorical features such as 'service' had high cardinality (70+ unique values), necessitating grouping of rare services into an "other" category.
- Several features were identified as redundant or having near-zero variance and were marked for removal.
- We visualized class separability and determined that while some attacks are distinct, many overlap with normal traffic, motivating the use of robust anomaly detection algorithms.

PHASE 2: MODELING AND EVALUATION
In 'notebooks/modeling_unsupervised.ipynb', we implemented and compared three unsupervised learning algorithms:
1. Isolation Forest
2. Local Outlier Factor (LOF)
3. Autoencoder (Neural Network)

The models were trained exclusively on "Normal" traffic to learn the baseline of legitimate behavior. We evaluated them using a test set containing both normal and attack traffic. 

Outcomes:
- Isolation Forest was selected as the best-performing model.
- It achieved a ROC-AUC of approximately 0.95 and high Recall at low False Positive Rates (FPR), making it suitable for a security context where missing attacks is costly but false alarms must be manageable.
- We defined thresholds (T_low, T_high) to categorize traffic as Normal, Suspicious, or Anomaly.
- The final model pipeline (preprocessing + model) was serialized and saved as 'artifacts/isolation_forest.joblib' and 'artifacts/preprocessor.joblib'.

PHASE 3: BACKEND DEVELOPMENT
We developed a REST API using FastAPI to serve the model.
- Created 'src/backend/main.py' to handle HTTP requests.
- Implemented 'src/backend/model_utils.py' to manage artifact loading, data preprocessing, and scoring logic.
- The system automatically handles missing input fields by defaulting them to zero or "other", ensuring robustness.
- Logging mechanisms were built to save every prediction request and user feedback to CSV files in 'src/backend/logs/', enabling future model retraining and monitoring.
- During development, a version mismatch issue with 'scikit-learn' caused artifact loading failures. This was resolved by creating a retraining script ('src/scripts/retrain.py') to regenerate compatible artifacts within the current environment.

PHASE 4: FRONTEND DEVELOPMENT
A lightweight, single-page web interface was built using HTML, CSS, and vanilla JavaScript.
- File: 'src/frontend/index.html'.
- The UI provides a form for users to input 12 key network features (e.g., Protocol, Service, Bytes, Duration).
- It communicates with the backend API to fetch anomaly scores and displays them in a clear Result Card.
- The interface includes "Mark Correct" and "Mark Incorrect" buttons to collect human feedback.

PHASE 5: INTEGRATION AND TESTING
We integrated the frontend and backend, configuring the FastAPI server to hose the static HTML files.
- Comprehensive testing was performed using a set of confirmed test cases ('data/processed/ui_test_cases.json') extracted from the original test dataset.
- We verified that the system correctly identifies known Normal traffic as "Normal" and known Attacks as "Anomaly".
- Edge cases, such as missing columns in the input JSON, were handled and verified to return successful predictions instead of errors.

PHASE 6: DOCUMENTATION AND DELIVERY
Finally, we consolidated all documentation.
- Created a 'README.txt' with simple instructions on how to run the system.
- Summarized model performance in 'reports/model_choice_summary.txt'.
- Documented EDA findings in 'reports/EDA_summary.txt'.
- All Markdown files were converted to plain text as per user preference to ensure simplicity and accessibility.

CONCLUSION
The system is now fully operational. It provides a user-friendly interface for detecting network anomalies, backed by a robust Isolation Forest model, with logging infrastructure to support continuous improvement.
