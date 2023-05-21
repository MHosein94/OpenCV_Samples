import numpy as np
import cv2
from tensorflow.keras.utils import img_to_array
from keras.models import model_from_json

from PIL import Image, ImageDraw
import gc

def detectFace(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    face_cascade = cv2.CascadeClassifier('Haar Cascade Files/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(frame_gray)
    faceROIs = []
    x_faces, y_faces, w_faces, h_faces = [], [], [], []
    for (x_face,y_face,w_face,h_face) in faces:
        #face
        center = (x_face + w_face//2, y_face + h_face//2)
        cv2.ellipse(frame, center, (w_face//2, h_face//2), angle=0, startAngle=0, endAngle=360, color=(255,200,45), thickness=5)
        faceROIs.append(frame_gray[x_face:x_face+h_face, y_face:y_face+h_face])   # Crop Face
        x_faces.append(x_face)
        y_faces.append(y_face)
        w_faces.append(w_face)
        h_faces.append(h_face)
    return (faceROIs, x_faces, y_faces, w_faces, h_faces)

video = cv2.VideoCapture(0)
model = model_from_json(open('Facial Expression/facial_expression_model_structure.json', 'r').read())
model.load_weights('Facial Expression/facial_expression_model_weights.h5')

emotions = ['Angry', 'Disgusted', 'Afraid', 'Happy', 'Sad', 'Suprised', 'Neutral']

while (True):
    _, frame = video.read()
    frame = cv2.flip(frame, 1)
    (faceROIs, xs, ys, ws, hs) = detectFace(frame)
    for (faceROI, x, y) in zip(faceROIs, xs, ys):
        try:
            faceROI = cv2.resize(faceROI, (48,48))

            img_pixels = img_to_array(faceROI)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255  # Normalize to [0,1]
            
            prediction = model.predict(img_pixels)
            predicted_emo = emotions[np.argmax(prediction)]

            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            draw.text((x, y), predicted_emo, fill=(50,50,250))
            frame = np.array(img_pil)
        except:
            pass

    cv2.imshow('Facial Expression', frame)
    gc.collect()


    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()