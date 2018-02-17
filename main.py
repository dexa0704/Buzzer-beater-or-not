import cv2
import numpy as np
import neuralNetwork


def checkShotTime(imgPath):
    '''

    Function for checking time on table semaphore. If zero on table, shot isn't valid
    Number recognitions with MNIST

    :param imgPath: relative path for image for checking time on table
    :return: boolen represent success, true -> valid shot, false -> invalid

    '''



    '''
    results = mnist(imgPath)
    if results[-1] == 0 
        return False
    '''
    print(imgPath)

    return True


def checkBallHandContact(imgPath):
    '''
    Function for detect ball on picture and try to find contact between ball and players hand
    Detection with open cv and numpy

    :param imgPath: relative path for image for checking time on table
    :return: boolen represent contact between hand and ball, true -> with contact, false -> without contact
    '''

    im = cv2.imread('../../test1.jpg')

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

    print(imgPath)

    return False


if __name__ == "__main__":
    print("Welcome to buzzer beater application :)\n\n")

    path = "dejan" #input("Insert relative image path for validation: ")

    contact = checkBallHandContact(path)
    tableSemaphore = checkShotTime(path)

    if contact is False or (contact is True and tableSemaphore is True):
        print("\n\nThis is VALID shot !")
    else:
        print("\n\nThis shot is INVALID !!!")
