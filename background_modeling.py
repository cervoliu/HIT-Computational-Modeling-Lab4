import numpy as np
import cv2

video = cv2.VideoCapture('.\\video\\test.avi')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

back = cv2.createBackgroundSubtractorMOG2() #MOG : Mixture Of Gaussian 

#read every frame
while True:
    ret, frame = video.read()
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

    k = cv2.waitKey(100) & 0xff
    if k == 27: #27 for ESC
        break

video.release()
cv2.destroyAllWindows()