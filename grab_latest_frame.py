import requests
import os

try:
    print("Fetching stream for current view...")
    res = requests.get("http://127.0.0.1:5000/api/v1/ai-identify/stream", stream=True, timeout=5)
    bytes_data = b''
    for chunk in res.iter_content(chunk_size=1024):
        bytes_data += chunk
        a = bytes_data.find(b'\xff\xd8')
        b = bytes_data.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            os.makedirs('captures', exist_ok=True)
            with open('captures/current_view.jpg', 'wb') as f:
                f.write(jpg)
            print("Saved captures/current_view.jpg")
            break
except Exception as e:
    print(f"Error: {e}")
