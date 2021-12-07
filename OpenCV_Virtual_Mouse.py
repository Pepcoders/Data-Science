import cv2 as cv
import mediapipe as mp
import autopy
import numpy as np
import math
import mouse


mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
tipIds = [4, 8, 12, 16, 20]
prevX, prevY = 0, 0
currX, currY = 0, 0


cap = cv.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 520)
widthScr, heighScr = autopy.screen.size()
smooth = 5
while True:
    success, img = cap.read()
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    processed_img = hands.process(img)
    if processed_img.multi_hand_landmarks:
        for handlndm in processed_img.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handlndm, mpHands.HAND_CONNECTIONS)

    # coordinates, box = detector.coordinates(img)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    xList = []
    yList = []
    bbox = []
    listLM = []
    if processed_img.multi_hand_landmarks:
        hand = processed_img.multi_hand_landmarks[0]  # handnumber
        for id, landmark in enumerate(hand.landmark):
            height, width, _ = img.shape
            cx, cy = int(landmark.x * width), int(landmark.y * height)
            xList.append(cx)
            yList.append(cy)
            listLM.append([id, cx, cy])
            cv.circle(img, (cx, cy), 15, (243, 59, 14), cv.FILLED)

        xmin, xmax = min(xList), max(xList)
        ymin, ymax = min(yList), max(yList)
        bbox = xmin, ymin, xmax, ymax
        cv.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)
    if len(listLM) != 0:
        xm, ym = listLM[8][1:]
        xc, yc = listLM[4][1:]
        fingers = []
        if listLM[tipIds[0]][1] < listLM[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            if listLM[tipIds[id]][2] < listLM[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        if fingers[1] == 1:
            x3 = np.interp(xm, (0, 640), (0, widthScr))
            y3 = np.interp(ym, (0, 480), (0, heighScr))
            currX = prevX + (x3 - prevX) / smooth
            currY = prevY + (y3 - prevY) / smooth
            autopy.mouse.move(widthScr - currX, currY)
            prevX, prevY = currX, currY
        if fingers[1] == 1 and fingers[0] == 1:
            x1, y1 = listLM[4][1:]
            x2, y2 = listLM[5][1:]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x1, y1), 15, (0, 0, 255), cv.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 45:
                cv.circle(img, (cx, cy), 15, (0, 255, 0), cv.FILLED)
                mouse.click()
        if fingers[2]==1 and fingers[1] == 1:
            x1, y1 = listLM[8][1:]
            x2, y2 = listLM[12][1:]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x1, y1), 15, (0, 0, 255), cv.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 45:
                cv.circle(img, (cx, cy), 15, (0, 255, 0), cv.FILLED)
                mouse.press()
            else:
                mouse.release()


    cv.imshow("Image", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break