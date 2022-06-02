from msilib.schema import Font
import cv2

import numpy as np

#detects faces
face_detector = cv2.CascadeClassifier(r'C:\\Users\\Tanusri.DESKTOP-LLOE865\\Documents\\haarcascade_frontalface_default.xml')
#detects smiles
smile_detector = cv2.CascadeClassifier(r'C:\Users\Tanusri.DESKTOP-LLOE865\Documents\haarcascade_smile.xml')

#captures the webcam
webcam = cv2.VideoCapture(0)

#displays the frame
while True:
    frame_read, frame = webcam.read()

    if not frame_read:
        break

    #converts to gray colour
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = face_detector.detectMultiScale(gray_scale)

    for (x, y, z, a) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + z, y + a), (0, 255, 0), 2)

        the_face = frame[y:y+a, x:x+z]

        face_gray = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smile_coordinate = smile_detector.detectMultiScale(face_gray, scaleFactor = 1.7, minNeighbors = 20)

        #displays a red coloured rectangle around smiles
        #for (l, m , o, p) in smile_coordinate:
           # cv2.rectangle(the_face, (l, m), (l + o, m + p), (255, 0, 0), 2)
        
        if len(smile_coordinate) > 0:
            cv2.putText(frame, "Smiling", (x, y+a+40), fontScale = 2, fontFace = cv2.FONT_HERSHEY_COMPLEX, color = (0, 0, 255) )


    cv2.imshow("Smile detector", frame)

    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break

webcam.release()
cv2.destroyAllWindows()