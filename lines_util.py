import cv2



def drawLine(img, p1, p2, color = (0,0,0)):
    cv2.line(img, p1, p2, color, 1)