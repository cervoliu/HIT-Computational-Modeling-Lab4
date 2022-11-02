import numpy as np
import cv2
import matplotlib.pyplot as plt

video = cv2.VideoCapture('.\\video\\test.avi')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

back = cv2.createBackgroundSubtractorMOG2() #MOG : Mixture Of Gaussian 

X = 200
Y = 300

pixelR = []
pixelG = []
pixelB = []
N = []

#read every frame
num = 0 # number of frame
while True:
    ret, frame = video.read()
    if frame is None: #end of video
        break

    num += 1
    N.append(num)
    pixelB.append(frame[X][Y][0])
    pixelG.append(frame[X][Y][1])
    pixelR.append(frame[X][Y][2])

    img = back.apply(frame)

    img_close = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(img_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        length = cv2.arcLength(cnt, True)
        if length > 188:
            x, y, w, h = cv2.boundingRect(cnt)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    cv2.imshow('frame', frame)
    cv2.imshow('img', img)

    k = cv2.waitKey(1) & 0xff
    if k == 27: #27 for ESC, press [ESC] to quit
        break

plt.subplot(3, 1, 1)
plt.title("RGB of pixel({0},{1})".format(X, Y))
plt.xlabel('frame')
plt.ylabel('B')
plt.plot(N, pixelB, 'b')

plt.subplot(3, 1, 2)
plt.xlabel('frame')
plt.ylabel('G')
plt.plot(N, pixelG, 'g')

plt.subplot(3, 1, 3)
plt.xlabel('frame')
plt.ylabel('R')
plt.plot(N, pixelR, 'r')

plt.show()

video.release()
cv2.destroyAllWindows()