import requests

base_url = "http://localhost:8080"

ping_resp = requests.get(f"{base_url}/ping")
print("Ping:", ping_resp.json())

payload = {"features": [5.1, 3.5, 1.4, 0.2]}
pred_resp = requests.post(f"{base_url}/invocations", json=payload)
print("Prediction:", pred_resp.json())