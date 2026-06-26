import cv2
import threading
import time

def run():
    print("Background thread starting...")
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    print('opened:', cap.isOpened())
    if cap.isOpened():
        r, f = cap.read()
        print('read:', r)
    cap.release()
    print("Background thread finished.")

t = threading.Thread(target=run)
t.start()
t.join(timeout=10)
print("Main thread done.")
