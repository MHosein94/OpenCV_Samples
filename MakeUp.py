import cv2
import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import time

video = cv2.VideoCapture(0)

process_this_time = True  # for faster processing, we process only odd frames
while (True):
    ret, frame = video.read()

    if process_this_time:
        face_landmarks = face_recognition.face_landmarks(frame)
        rgb_small_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        pil_image = Image.fromarray(rgb_small_frame)
        drawer = ImageDraw.Draw(pil_image, mode = 'RGBA')
        for face_lnd in face_landmarks:
            drawer.polygon(face_lnd['left_eyebrow'], fill = (43, 21, 14, 230))
            drawer.polygon(face_lnd['right_eyebrow'], fill = (43, 21, 14, 230))
            #
            drawer.polygon(face_lnd['left_eye'], fill = (250, 20, 20, 50))
            drawer.polygon(face_lnd['right_eye'], fill = (250, 20, 20, 50))
            #
            drawer.line(face_lnd['left_eye'] + [face_lnd['left_eye'][0]], 
                        fill = (0, 0, 0, 200), width = 3)
            drawer.line(face_lnd['right_eye'] + [face_lnd['right_eye'][0]], 
                        fill = (0,0,0, 200), width = 3)
            #
            drawer.polygon(face_lnd['bottom_lip'], fill = (205, 20,20, 200))
            drawer.polygon(face_lnd['top_lip'], fill = (205, 20,20, 200))
        del drawer
    
    process_this_time = not process_this_time
    final_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGR)
    cv2.imshow('Video', final_frame)
    
    # Hit 'q' on the keyboard to quit!
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

# Release handle to the webcam
video.release()
cv2.destroyAllWindows()
