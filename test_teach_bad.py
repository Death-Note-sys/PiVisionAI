import cv2
import requests

print("Fetching stream for Bad object...")
res = requests.get("http://127.0.0.1:5000/api/v1/ai-identify/stream", stream=True)
bytes_data = b''
for chunk in res.iter_content(chunk_size=1024):
    bytes_data += chunk
    a = bytes_data.find(b'\xff\xd8')
    b = bytes_data.find(b'\xff\xd9')
    if a != -1 and b != -1:
        jpg = bytes_data[a:b+2]
        with open('captures/bad_frame.jpg', 'wb') as f:
            f.write(jpg)
        print("Frame saved to captures/bad_frame.jpg")
        break

# Same coordinates used for good
x, y, bw, bh = 320, 180, 640, 360
payload = {"x": x, "y": y, "w": bw, "h": bh}

print(f"Teaching 'Bad' with coordinates: x={x}, y={y}, w={bw}, h={bh}")
res = requests.post(
    "http://127.0.0.1:5000/api/v1/ai-identify/teach-bad",
    json=payload
)
print("Teach Bad response:", res.status_code, res.text)

# Check status
status_res = requests.get("http://127.0.0.1:5000/api/v1/ai-identify/status")
print("Status response:", status_res.status_code, status_res.text)
