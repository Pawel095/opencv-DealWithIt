import numpy as np
import cv2
from mathUtils import get_dist

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
camera = cv2.VideoCapture(0)
while True:
    ret, img = camera.read()
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_grey = grey[y:y + h, x:x + w]
        face_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(face_grey, 1.3, 4)
        eyes_list = list()
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            if len(eyes) == 2:
                oko = list()
                oko.append(ex + ew / 2)
                oko.append(ey + eh / 2)
                eyes_list.append(oko)

        if len(eyes_list)==2:
            pass


    cv2.imshow("img", img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
camera.release()
cv2.destroyAllWindows()
