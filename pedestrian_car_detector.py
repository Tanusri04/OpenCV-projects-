import cv2
import numpy as np

#specifying the path to the video 
vid_file = r"C:\Users\Tanusri.DESKTOP-LLOE865\Downloads\videoplayback (1).mp4"

#captures the video file
video = cv2.VideoCapture(vid_file)

#xml file of pre-trained car and pedestrian classifier 
classifier_path = cv2.CascadeClassifier(r"C:\Users\Tanusri.DESKTOP-LLOE865\Downloads\37e1e0af2bf8965e8058a9dfa3285bc6-e690cef3fb3ede5b869c3969cd6a9c5735d4ec7b\cars.xml")
ped_classifier = cv2.CascadeClassifier(r"C:\Users\Tanusri.DESKTOP-LLOE865\Documents\haarcascade_fullbody.xml")

while True:
    successful_frame, frame = video.read()

    if successful_frame:
        gray_vid = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    #detects cars
    detector = classifier_path.detectMultiScale(gray_vid)
    ped_detector = ped_classifier.detectMultiScale(gray_vid)

    #draws green rectangles for cars
    for(x, y, w, h) in detector:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    #draws yellow rectangles for pedestrians
    for(x, y, w, h) in ped_detector:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)    
    
    #displays the image file
    cv2.imshow("Car and pedestrian detector", frame)

    key = cv2.waitKey(1)
    #if 'Q' or 'q' is pressed, the window is quit 
    if key == 81 or key == 113:
        break

vid_file.release()
