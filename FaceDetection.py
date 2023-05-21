import cv2

def detectFace(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    face_cascade = cv2.CascadeClassifier('Haar Cascade Files/haarcascade_frontalface_alt.xml')
    eyes_cascade = cv2.CascadeClassifier('Haar Cascade Files/haarcascade_eye.xml')
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x_face,y_face,w_face,h_face) in faces:
        # face
        center = (x_face + w_face//2, y_face + h_face//2)
        cv2.ellipse(frame, center, (w_face//2, h_face//2), angle=0, startAngle=0, endAngle=360, color=(255,200,45), thickness=6)
        # Region of Interest
        faceROI = frame_gray[y_face:y_face+h_face, x_face:x_face+w_face]
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x_eye, y_eye, w_eye, h_eye) in eyes:
            center = (x_face+x_eye + w_eye//2, y_face+y_eye + h_eye//2)
            cv2.ellipse(frame, center, (w_eye//4, h_eye//4), 0, 0, 360, (80,55,80), thickness=4)

video = cv2.VideoCapture(0)

while (True):
    _, frame = video.read()
    frame = cv2.flip(frame, 1)
    detectFace(frame)
    cv2.imshow('Detected ROI', frame)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()