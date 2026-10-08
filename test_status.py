import requests
import json
import time

requests.post("http://127.0.0.1:5000/api/v1/ai-identify/start")
time.sleep(1) # wait for process loop
res = requests.get("http://127.0.0.1:5000/api/v1/ai-identify/status")
print(res.status_code, json.dumps(res.json(), indent=2))
