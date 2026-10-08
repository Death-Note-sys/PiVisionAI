import cv2

img = cv2.imread('captures/current_view.jpg')
# Draw a rectangle to check the crop
x, y, w, h = 100, 320, 180, 320
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
cv2.imwrite('captures/test_crop.jpg', img)
print("Saved test_crop.jpg")
