import json
import cv2
import numpy as np

with open('app/modules/ai_identify/controller.py', 'r') as f:
    code = f.read()

# Replace __init__ to load data
init_patch = '''
        # Load settings
        with self._lock:
            saved_settings = self.load_settings()
            if saved_settings:
                self.settings.update(saved_settings)
            
            # Load persistence
            data = self.load_data()
            if data and "teach_status" in data:
                self.teach_status = data.get("teach_status", "Untaught")
                for i in range(data.get("good_count", 0)):
                    img = self.load_image(f"good_{i}.png")
                    if img is not None:
                        kp, des = self.orb.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
                        self.good_references.append({"image": img, "keypoints": kp, "descriptors": des})
                for i in range(data.get("bad_count", 0)):
                    img = self.load_image(f"bad_{i}.png")
                    if img is not None:
                        kp, des = self.orb.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
                        self.bad_references.append({"image": img, "keypoints": kp, "descriptors": des})
'''
if 'self.load_data()' not in code:
    code = code.replace('# Load settings\n        with self._lock:', init_patch)

def patch_teach(code, kind):
    target = f'self.{kind}_references.append({{"image": crop, "keypoints": kp, "descriptors": des}})'
    replacement = target + f'''
            self.save_image("{kind}_" + str(len(self.{kind}_references)-1) + ".png", crop)
            self.save_data({{"teach_status": self.teach_status, "good_count": len(self.good_references), "bad_count": len(self.bad_references)}})
'''
    return code.replace(target, replacement)

if 'self.save_image("good_' not in code:
    code = patch_teach(code, 'good')
if 'self.save_image("bad_' not in code:
    code = patch_teach(code, 'bad')

with open('app/modules/ai_identify/controller.py', 'w') as f:
    f.write(code)

