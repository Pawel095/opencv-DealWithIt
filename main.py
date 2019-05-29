import numpy as np
import cv2
import math

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
camera = cv2.VideoCapture(0)
while True:
    ret, img = camera.read()
    grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey,1.3,5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("img", img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
camera.release()
cv2.destroyAllWindows()
