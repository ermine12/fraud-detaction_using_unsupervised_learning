import requests
import json
import time

URL = "http://127.0.0.1:8000"

def test_predict():
    print("Testing /predict...")
    # Load test cases
    try:
        with open(r"c:\Users\ELITEBOOK\Documents\Projects\4th_2018_Class_projects\ML\fraud-detaction_using_unsupervised_learning\data\processed\ui_test_cases.json", "r") as f:
            cases = json.load(f)
    except FileNotFoundError:
        print("ui_test_cases.json not found. Using dummy data.")
        cases = [{"duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF", "src_bytes": 123, "dst_bytes": 456}]

    for i, case in enumerate(cases[:2]): # Test first 2
        # case might have '_expected_label'
        payload = {k: v for k, v in case.items() if not k.startswith('_')}
        
        try:
            resp = requests.post(f"{URL}/predict", json=payload)
            print(f"Case {i}: Status {resp.status_code}")
            if resp.status_code == 200:
                print("Response:", json.dumps(resp.json(), indent=2))
            else:
                print("Error:", resp.text)
        except Exception as e:
            print(f"Request failed: {e}")

def test_feedback():
    print("\nTesting /feedback...")
    feedback_data = {
        "request_id": "test_request_123",
        "feedback_label": "correct",
        "actual_label": "Normal"
    }
    try:
        resp = requests.post(f"{URL}/feedback", json=feedback_data)
        print(f"Feedback Status: {resp.status_code}")
        print("Response:", resp.json())
    except Exception as e:
        print(f"Feedback failed: {e}")

if __name__ == "__main__":
    # Wait for server to be ready?
    time.sleep(2) 
    test_predict()
    test_feedback()
