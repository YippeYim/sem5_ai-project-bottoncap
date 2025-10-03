import cv2
img = cv2.imread("/home/pi/Desktop/test/test_cap.JPG")
cv2.imshow("Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()