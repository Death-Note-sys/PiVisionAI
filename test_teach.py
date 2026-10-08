import cv2
import requests
import json

img = cv2.imread('captures/good_frame.jpg')
if img is None:
    print("Failed to load captures/good_frame.jpg")
    exit(1)

h, w = img.shape[:2]
bw, bh = w // 2, h // 2
x = (w - bw) // 2
y = (h - bh) // 2

print(f"Teaching 'Good' with coordinates: x={x}, y={y}, w={bw}, h={bh}")

payload = {"x": x, "y": y, "w": bw, "h": bh}
res = requests.post(
    "http://127.0.0.1:5000/api/v1/ai-identify/teach-good",
    json=payload
)
print("Teach Good response:", res.status_code, res.text)
