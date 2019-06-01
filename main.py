import numpy as np
import cv2
import math
import imutils
from mathUtils import get_dist, approach


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


class overlay:
    maxFramesOutOfImage = 5
    framesOutOfImage = 0
    recentlyDrawn = True
    #                  x, y
    desiredPosition = [0, 0]
    currentPosition = [300, -500]

    #            rot, scl
    desiredWarp = [0, 1]
    currentWarp = [0, 1]

    def __init__(self):
        self.img = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)
        self.img = cv2.resize(self.img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    def draw(self, background):
        self.recentlyDrawn = True

        delta = (self.desiredWarp[0] - self.currentWarp[0]) * 1/4
        self.currentWarp[0]+= delta

        delta = (self.desiredWarp[1] - self.currentWarp[1]) * 1 / 4
        self.currentWarp[1]+= delta

        delta = (self.desiredPosition[0] - self.currentPosition[0]) * 3/5
        self.currentPosition[0] += int(delta)

        delta = (self.desiredPosition[1] - self.currentPosition[1]) * 3/5
        self.currentPosition[1] += int(delta)

        scaledGlasses = cv2.resize(self.img, None, fx=self.currentWarp[1], fy=self.currentWarp[1],
                                   interpolation=cv2.INTER_LINEAR)
        rotatedGlasses = imutils.rotate_bound(scaledGlasses, self.currentWarp[0])
        add_transparent_image(background, rotatedGlasses, int(self.currentPosition[1]), int(self.currentPosition[0]))
        cv2.line(background,tuple(self.currentPosition),tuple(self.desiredPosition),(255,255,0),4)

    def isUnused(self):
        if self.framesOutOfImage < self.maxFramesOutOfImage:
            if not self.recentlyDrawn:
                self.framesOutOfImage += 1

            self.recentlyDrawn = False
            return False
        else:
            return True

    def setattr(self, pos, warp):
        self.desiredPosition = pos
        self.desiredWarp = warp


glasses = list()
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
camera = cv2.VideoCapture(0)
while True:
    # remove all out of frame glasses
    for i, g in enumerate(glasses):
        if g.isUnused():
            print("isRemovable True, deleting " + i.__str__())
            glasses.pop(i)

    ret, img = camera.read()
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, 1.3, 5)
    # for each face
    for i, (x, y, w, h) in enumerate(faces):

        # if overlay not existing then create
        if not i < len(glasses):
            print("creating " + i.__str__())
            glasses.append(overlay())

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, i.__str__(), (x, y + 25), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))

        face_grey = grey[y:y + h, x:x + w]
        face_color = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(face_grey, 1.3, 4)
        eyes_list = list()
        for (ex, ey, ew, eh) in eyes:
            if len(eyes) == 2:
                oko = [ex + ew / 2, ey + eh / 2]
                eyes_list.append(oko)
                cv2.putText(face_color, i.__str__(), (int(oko[0]), int(oko[1])), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0))

        # jeżeli jedna para i różnica y jest poniżej 30.
        if len(eyes_list) == 2:
            deltaY = abs(eyes_list[0][1] - eyes_list[1][1])
            # print(deltaY)
            if deltaY < 30:
                scale = get_dist(eyes_list) / (225 * 0.7)
                tanTheta = (eyes_list[0][1] - eyes_list[1][1]) / (eyes_list[0][0] - eyes_list[1][0])
                angle = math.degrees(math.atan(tanTheta))

                center_x = (eyes_list[0][0] + eyes_list[1][0]) / 2
                center_y = (eyes_list[0][1] + eyes_list[1][1]) / 2
                newx = int(x + center_x)
                newy = int(y + center_y)
                glasses[i].setattr([newx, newy], [angle, scale])

                cv2.circle(img, (int(x + eyes_list[0][0]), int(y + eyes_list[0][1])), 20, (0, 255, 0), 1)
                cv2.circle(img, (int(x + eyes_list[1][0]), int(y + eyes_list[1][1])), 20, (0, 255, 0), 1)

        glasses[i].draw(img)

    cv2.imshow("img", img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
camera.release()
cv2.destroyAllWindows()
