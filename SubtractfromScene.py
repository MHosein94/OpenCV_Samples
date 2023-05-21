import cv2
import numpy as np
import time

time.sleep(3)
video = cv2.VideoCapture(0)
# Capture first frame
_, first_frame = video.read()
first_frame = np.array(cv2.flip(first_frame, 1), dtype='uint8')
time.sleep(3)

while(True):
    _, frame = video.read()
    frame = cv2.flip(frame, 1)

    sub = cv2.subtract(frame, first_frame)
    cv2.imshow('Subtracted', sub)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()