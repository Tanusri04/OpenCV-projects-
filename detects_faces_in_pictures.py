
#Code to detect faces using pictures
import cv2 
import numpy as np

#path to the image 
path = r"C:\Users\Tanusri.DESKTOP-LLOE865\Desktop\dr.png"

#an algorithm which is pre-trained on frontal face
trained_face = cv2.CascadeClassifier('C:\\Users\\Tanusri.DESKTOP-LLOE865\\Documents\\haarcascade_frontalface_default.xml') 

#the image to detect the face
img = cv2.imread(path) 

#changes the colour of picture to gray
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

#detect face
face_coordinates = trained_face.detectMultiScale(gray_img)

for (x, y, z, a) in face_coordinates:
    cv2.rectangle(img, (x, y), (x + z, y + a), (0, 255, 0), 2)


cv2.imshow('Face detector', img) 

#print(face_coordinates)
#print("Code completed")

cv2.waitKey(0)
cv2.destroyAllWindows()

