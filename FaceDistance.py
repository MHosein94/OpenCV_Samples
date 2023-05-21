import cv2
import face_recognition

video = cv2.VideoCapture(0)
process_this_time = True

while (True):
    if process_this_time:
        _, frame = video.read()
        frame_encodings = face_recognition.face_encodings(frame)
        face_distance = 0
        if len(frame_encodings) == 2: # Threre are two people in the picture
            face_distance = face_recognition.face_distance([frame_encodings[0]], frame_encodings[1])

            frame = cv2.putText(
                frame, 
                text=str(face_distance[0]), 
                org=(50, 50), 
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=2, color=(220, 25, 0), 
                thickness=2
                )

    process_this_time = not process_this_time
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break
    cv2.imshow('Face distance', frame)
video.release()
cv2.destroyAllWindows()
