import requests
import cv2
import numpy as np
import os

if not os.path.exists('captures'):
    os.makedirs('captures')

try:
    print("Starting AI Identify module...")
    res = requests.post("http://127.0.0.1:5000/api/v1/ai-identify/start")
    print("Start response:", res.status_code, res.text)
except requests.exceptions.ConnectionError:
    print("CONNECTION_REFUSED")
    exit(1)

print("Fetching stream...")
res = requests.get("http://127.0.0.1:5000/api/v1/ai-identify/stream", stream=True)
bytes_data = b''
for chunk in res.iter_content(chunk_size=1024):
    bytes_data += chunk
    a = bytes_data.find(b'\xff\xd8')
    b = bytes_data.find(b'\xff\xd9')
    if a != -1 and b != -1:
        jpg = bytes_data[a:b+2]
        
        with open('captures/good_frame.jpg', 'wb') as f:
            f.write(jpg)
        print("Frame saved to captures/good_frame.jpg")
        break
