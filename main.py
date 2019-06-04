import numpy as np
import cv2
import math
import imutils
from mathUtils import get_dist
import statistics
import random


def exists(o):
    try:
        o
    except NameError:
        return False
    else:
        return True


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
    maxFramesOutOfImage = 20
    framesOutOfImage = 0
    recentlyDrawn = True
    #                  x, y
    currentPosition = [300, -500]

    #            rot, scl
    currentWarp = [0, 0.3]
    def __init__(self,i):
        self.img = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)
        self.img = cv2.resize(self.img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        self.name = i

    def draw(self, background):
        self.recentlyDrawn = True

        scaledGlasses = cv2.resize(self.img, None, fx=self.currentWarp[1], fy=self.currentWarp[1],
                                   interpolation=cv2.INTER_LINEAR)
        rotatedGlasses = imutils.rotate_bound(scaledGlasses, self.currentWarp[0])
        add_transparent_image(background, rotatedGlasses, int(self.currentPosition[1]), int(self.currentPosition[0]))

    def isUnused(self):
        if self.framesOutOfImage < self.maxFramesOutOfImage:
            if not self.recentlyDrawn:
                self.framesOutOfImage += 1

            self.recentlyDrawn = False
            return False
        else:
            return True

    def setx(self, a):
        self.currentPosition[0]=statistics.mean((self.currentPosition[0],a))

    def sety(self, a):
        self.currentPosition[1]=statistics.mean((self.currentPosition[1],a))

    def setAngle(self, a):
        pass

    def setScale(self, a):
        pass

    def setattr(self, pos, warp):
        self.currentPosition[0] = statistics.mean((self.currentPosition[0], pos[0]))
        self.currentPosition[1] = statistics.mean((self.currentPosition[1], pos[1]))

        self.currentWarp=warp


glasses = {}

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
camera = cv2.VideoCapture(0)
while True:
    ret, img = camera.read()
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, 1.3, 5)

    # todo create overlay classes
    for i, face in enumerate(faces):
        if i >= len(glasses):
            print("creating "+i.__str__())
            glasses[i]=overlay(i)

    detectionResult = {}
    index =0
    for x, y, w, h in faces:

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

        detectionResult[index] = face
        index+=1

    calculationResult = {}
    for i, face in detectionResult.items():
        if "eyes" in face:
            scale = get_dist(face["eyes"]) / (225 * 0.7)
            tanTheta = (face["eyes"][0][1] - face["eyes"][1][1]) / (face["eyes"][0][0] - face["eyes"][1][0])
            angle = math.degrees(math.atan(tanTheta))
            targetx = int(face["face"][0] + (face["eyes"][0][0] + face["eyes"][1][0]) / 1.5)
            targety = int(face["face"][1] + (face["eyes"][0][1] + face["eyes"][1][1]) / 1.5)
            calculationResult[i] = {"pos": [targetx, targety], "warp": [angle, scale]}
        else:
            targetx = int(face["face"][0] + face["face"][2] / 2)
            targety = int(face["face"][1] + face["face"][3] / 2)
            calculationResult[i] = {"pos": [targetx, targety]}
    # return lista dictów z pos = [x,y] warp = [rot,scl]

    #przypisz overlay do face na podstawie indexów



    for i, face in calculationResult.items():
        minDist = 19823912481
        index = 0
        for j,glass in glasses.items():
            if get_dist((glass.currentPosition,face["pos"])) < minDist:
                minDist = get_dist((glass.currentPosition,face["pos"]))
                index = j
                print(glasses[j].name)
        current = glasses[index]
        print("selecting",current.name,minDist)


        if "warp" in face:
            current.setattr(face["pos"], face["warp"])
        else:
            current.setx(face["pos"][0])
            current.sety(face["pos"][1])

    if len(glasses.items()) >0 :
        glasses[0].draw(img)
    print("##########################################################")
    # todo remove usless uverlay classes
    removelist=[]
    for i,g in glasses.items():
        if g.isUnused():
            print("killing "+i.__str__())
            removelist.append(i)
    for i in removelist:
        glasses.pop(i)

    cv2.imshow("img", img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

camera.release()
cv2.destroyAllWindows()
