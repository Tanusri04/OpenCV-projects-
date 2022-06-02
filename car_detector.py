
#Code for detecting a picture of a car

import cv2

import numpy as np

classifier_path = cv2.CascadeClassifier(r"C:\Users\Tanusri.DESKTOP-LLOE865\Downloads\37e1e0af2bf8965e8058a9dfa3285bc6-e690cef3fb3ede5b869c3969cd6a9c5735d4ec7b\cars.xml")


#image of a car
img_file = r"C:\Users\Tanusri.DESKTOP-LLOE865\Desktop\picccc.png"

#reads the image file
img = cv2.imread(img_file)

#converts to gray scale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detects cars
detector = classifier_path.detectMultiScale(gray_img)

#draws rectangles around the detected cars
for(x, y , w, h) in detector:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)


#Display the image file
cv2.imshow("Car and pedestrian detector", img)

#does not close the window till a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()




  