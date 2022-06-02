import cv2 

from random import randrange

#detects faces
trained_face = cv2.CascadeClassifier('C:\\Users\\Tanusri.DESKTOP-LLOE865\\Documents\\haarcascade_frontalface_default.xml') 

#captures the webcam
webcam = cv2.VideoCapture(0)

#displays the frame
while True:
    frame_read, frame = webcam.read()
    
    #converts to gray 
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    #has the coordinates of the detected faces
    face_coordinates = trained_face.detectMultiScale(gray_img)

    #draws rectangles based on the coordinates
    for (x, y, z, a) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + z, y + a), (0, randrange(255), 0), 2)

    cv2.imshow('Face detector', frame) 

    key = cv2.waitKey(1)
    
    #if 'Q' or 'q' is pressed, the window is quit
    if key == 81 or key == 113:
        break

