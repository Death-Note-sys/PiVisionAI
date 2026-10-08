with open('app/modules/measurement/controller.py', 'r') as f:
    meas = f.read()

meas = meas.replace('rect = cv2.minAreaRect(c)', 'rect = cv2.minAreaRect(c)\n            x, y, _, _ = cv2.boundingRect(c)')

with open('app/modules/measurement/controller.py', 'w') as f:
    f.write(meas)
