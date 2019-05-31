import numpy as np
import cv2
import math
import imutils
from mathUtils import get_dist


def add_transparent_image(bg, ov, x, y):
    x -= int(ov.shape[0] / 2)
    y -= int(ov.shape[1] / 2)
    w = x + ov.shape[0]
    h = y + ov.shape[1]

    alpha_ov = ov[:, :, 3] / 255.0

    alpha_bg = 1.0 - alpha_ov

    for c in range(0, 3):
        try:
            bg[x:w, y:h, c] = (ov[:, :, c] * alpha_ov + bg[x:w, y:h, c] * alpha_bg)
        except:
            pass


glasses = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)
glasses = cv2.resize(glasses, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)

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

        if len(eyes_list) == 2:
            scale = get_dist(eyes_list) / (225*0.7)
            tanTheta = (eyes_list[0][1] - eyes_list[1][1]) / (eyes_list[0][0] - eyes_list[1][0])
            angle = math.degrees(math.atan(tanTheta))

            scaledGlasses = cv2.resize(glasses, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            rotatedGlasses = imutils.rotate_bound(scaledGlasses, angle)

            center_x = (eyes_list[0][0] + eyes_list[1][0])/2
            center_y = (eyes_list[0][1] + eyes_list[1][1])/2
            add_transparent_image(img,rotatedGlasses, int(y + center_y), int(x + center_x))

    cv2.imshow("img", img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
camera.release()
cv2.destroyAllWindows()
