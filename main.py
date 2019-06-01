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
    desiredWarp = [0, 0.5]
    currentWarp = [0, 1]

    def __init__(self):
        self.img = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)
        self.img = cv2.resize(self.img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    def draw(self, background):
        self.recentlyDrawn = True

        delta = (self.desiredWarp[0] - self.currentWarp[0]) * 1 /4
        self.currentWarp[0] += delta

        delta = (self.desiredWarp[1] - self.currentWarp[1]) * 1/4
        self.currentWarp[1] += delta

        delta = (self.desiredPosition[0] - self.currentPosition[0]) * 1/4
        self.currentPosition[0] += int(delta)

        delta = (self.desiredPosition[1] - self.currentPosition[1]) * 1/4
        self.currentPosition[1] += int(delta)

        scaledGlasses = cv2.resize(self.img, None, fx=self.currentWarp[1], fy=self.currentWarp[1],
                                   interpolation=cv2.INTER_LINEAR)
        rotatedGlasses = imutils.rotate_bound(scaledGlasses, self.currentWarp[0])
        add_transparent_image(background, rotatedGlasses, int(self.currentPosition[1]), int(self.currentPosition[0]))
        cv2.line(background, tuple(self.currentPosition), tuple(self.desiredPosition), (255, 255, 0), 4)

    def isUnused(self):
        if self.framesOutOfImage < self.maxFramesOutOfImage:
            if not self.recentlyDrawn:
                self.framesOutOfImage += 1

            self.recentlyDrawn = False
            return False
        else:
            return True

    def setx(self, a):
        self.desiredPosition[0] = a

    def sety(self, a):
        self.desiredPosition[1] = a

    def setAngle(self, a):
        self.desiredWarp[0] = a

    def setScale(self, a):
        self.desiredWarp[1] = a

    def setattr(self, pos, warp):
        self.desiredPosition = pos
        self.desiredWarp = warp
        self.currentPosition = pos
        self.currentWarp = warp


glasses = [overlay(), overlay(), overlay()]

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
camera = cv2.VideoCapture(0)
img = cv2.imread("myUglyFace.jpg")
while True:
    ret, img = camera.read()
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, 1.3, 5)

    detectionResult = []
    for i, (x, y, w, h) in enumerate(faces):
        face = {"face": (x, y, w, h)}
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, i.__str__(), (x, y + h), cv2.FONT_HERSHEY_PLAIN, 12, (255, 0, 0), 5)

        face_grey = grey[y:y + h, x:x + w]
        face_img = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(face_grey, 1.3, 4)
        if len(eyes) == 2:
            ax, ay, aw, ah = eyes[0]
            bx, by, bw, bh = eyes[1]
            deltaY = abs(ay - by)
            if deltaY < 30:
                face["eyes"] = ((ax, ay, aw, ah), (bx, by, bw, bh))
                cv2.rectangle(face_img, (ax, ay), (ax + aw, ay + ah), (255, 255, 255), 4)
                cv2.rectangle(face_img, (bx, by), (bx + bw, by + bh), (255, 255, 255), 4)
                cv2.putText(face_img, '0', (ax, ay + ah), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 2)
                cv2.putText(face_img, '1', (bx, by + bh), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 2)

        detectionResult.append(face)

    calculationResult = []
    for i, face in enumerate(detectionResult):
        if "eyes" in face:
            scale = get_dist(face["eyes"]) / (225 * 0.7)
            tanTheta = (face["eyes"][0][1] - face["eyes"][1][1]) / (face["eyes"][0][0] - face["eyes"][1][0])
            angle = math.degrees(math.atan(tanTheta))
            targetx = int(face["face"][0] + (face["eyes"][0][0] + face["eyes"][1][0]) / 1.5)
            targety = int(face["face"][1] + (face["eyes"][0][1] + face["eyes"][1][1]) / 1.5)
            calculationResult.append({"pos": [targetx, targety], "warp": [angle, scale]})
        else:
            targetx = int(face["face"][0] + face["face"][2] / 2)
            targety = int(face["face"][1] + face["face"][3] / 2)
            calculationResult.append({"pos": [targetx, targety]})

    for i, face in enumerate(calculationResult):
        current = glasses[i]
        # TODO tutaj pilnowanie co jest w dict a czego nie ma
        if "warp" in face:
            current.setattr(face["pos"],face["warp"])
        else:
            current.setx(face["pos"][0])
            current.sety(face["pos"][1])

        current.draw(img)

    cv2.imshow("img", img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

camera.release()
cv2.destroyAllWindows()
