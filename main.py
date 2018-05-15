import cv2
import numpy as np
import neuralNetwork as nn
from matplotlib import pyplot as plt


def shotClockViolation(imgPath):
    '''

    Function for checking time on table semaphore. If zero on table, shot isn't valid. Checking specific circle on the
    semaphore. If that circle exist clock is on zero, if circle doesn't exist time still counting ...


    :param imgPath: relative path for image for checking time on table
    :return: boolen represent presents of violation, true -> there is a violation, false -> no violation

    '''

    im = cv2.imread('../../oneSecLeft.jpg')
    print(im.shape)
    # prva koord je Y a druga X, X ide lepo sa leva na desno od 0.0, a y ide odozgo na dole 0,0 mu je gore !
    crop_img = im[870:1000, 1228:1450] #- ovo je za semaphoe four.. 3:09 ta tura slika
    #crop_img = im[980:1100, 1128:1300] # - ovo je sa 5:57 tura slika

    plt.imshow(crop_img, interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

    bilateral_filtered_image = cv2.bilateralFilter(crop_img, 5, 175, 175)

    gray = cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    edge_detected_image = cv2.Canny(gray, 75, 200)
    cv2.imshow('Edge', edge_detected_image)
    cv2.waitKey(0)

    rows = gray.shape[0]

    _, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #circles = cv2.HoughCircles(edge_detected_image, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=30, minRadius=1, maxRadius=30)

    #print(circles)
    #print(len(contours))

    if len(contours) != 0:
        print("\n\nShot clock violation !!! \n\n")
        return True
    else:
        print("\n\nStill have time...\n\n")
        return False


def checkBallHandContact(imgPath):
    '''
    Function for detect ball on picture and try to find contact between ball and players hand
    Detection with open cv and numpy

    :param imgPath: relative path for image for checking ball contact with hand
    :return: boolean represent contact between hand and ball, true -> with contact, false -> without contact
    '''

    '''
    # convert to HSV space
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # take only the orange, highly saturated, and bright parts
    im_hsv = cv2.inRange(im_hsv, (7, 103, 103), (11, 255, 255))

    # To show the detected orange parts:
    im_orange = im.copy()
    im_orange[im_hsv == 0] = 0
    # cv2.imshow('im_orange',im_orange)

    # Perform opening to remove smaller elements
    element = np.ones((5, 5)).astype(np.uint8)
    im_hsv = cv2.erode(im_hsv, element)
    im_hsv = cv2.dilate(im_hsv, element)

    points = np.dstack(np.where(im_hsv > 0)).astype(np.float32)
    # fit a bounding circle to the orange points
    center, radius = cv2.minEnclosingCircle(points)
    # draw this circle
    cv2.circle(im, (int(center[1]), int(center[0])), int(radius), (255, 0, 0), thickness=3)

    out = np.vstack([im_orange, im])
    cv2.imwrite('out.png', out)
    '''

    #print(imgPath)

    return True


if __name__ == "__main__":

    print("Welcome to buzzer beater application :)\n\n")

    path = "dejan" #input("Insert relative image path for validation: ")

    contact = checkBallHandContact(path)
    violation = shotClockViolation(path)

    if contact is False:
        print("\n\nThis is VALID shot :) !")
    elif violation is True:
        print("\n\nThis shot is INVALID !!!")
    else:
        print("\n\nThis is VALID shot :) !")
