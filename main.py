import cv2
import os
import glob
import numpy as np
from matplotlib import pyplot as plt


def getDistance(pixel, refcolor):

    return abs((pixel[0]-refcolor[0]) + (pixel[1]-refcolor[1]) + (pixel[2]-refcolor[2]))


def shotClockViolation(imgPath):
    '''

    Function for checking time on table semaphore. If zero on table, shot isn't valid. Checking specific circle on the
    semaphore. If that circle exist clock is on zero, if circle doesn't exist time still counting ...


    :param imgPath: relative path for image for checking time on table
    :return: boolen represent presents of violation, true -> there is a violation, false -> no violation

    '''

    #im = cv2.imread('../../MyDataset/Bacanje/Poza1/IMAG0088_BURST001.jpg')
    im = cv2.imread(imgPath)
    #print(im.shape)
    # prva koord je Y a druga X, X ide lepo sa leva na desno od 0.0, a y ide odozgo na dole 0,0 mu je gore !

    crop_img = im[690:810, 1900:2080]

    
    #### Slobodno bacanje !

    #crop_img = im[880:950, 1020:1110] #  - Poza1
    #crop_img = im[1000:1080, 1128:1280] # - ovo je sa 5:57 tura slika - Poza2
    #crop_img = im[690:830, 1108:1225] # - poza 3
    #crop_img = im[870:980, 1228:1420] #- ovo je za semaphoe four.. 3:09 ta tura slika - Poza4 - jedina mix poza
    #crop_img = im[693:760, 1170:1300]  # - poza 5


    ### Desnih 45

    #crop_img = im[580:680, 1910:1940]  - Poza1
    #crop_img = im[690:760, 1935:2050] - Poza2
    #crop_img = im[650:710, 1780:1910] - Poza3
    #crop_img = im[570:690, 1880:2030] - Poza4
    #crop_img = im[690:810, 1900:2080] - Poza5

    plt.imshow(crop_img, interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

    bilateral_filtered_image = cv2.bilateralFilter(crop_img, 5, 175, 175)

    gray = cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    edge_detected_image = cv2.Canny(gray, 75, 200)
    #cv2.imshow('Edge', edge_detected_image)
    #cv2.waitKey(0)

    rows = gray.shape[0]

    _, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #circles = cv2.HoughCircles(edge_detected_image, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=30, minRadius=1, maxRadius=30)

    #bounding areas . .

    #print(circles)
    #print(len(contours))

    if len(contours) != 0:
        print("\n\nShot clock violation !!!")
        return True
    else:
        print("\n\nStill have time...")
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

    imgSome = '../../MyDataset/Bacanje/Poza1/IMAG0088_BURST001.jpg'

    im = cv2.imread(imgSome)
    crop_img = im[950:, ]

    im_hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    low_range = np.array([0, 10, 60])# 0, 10, 60   128,0,0
    upper_range = np.array([20, 150, 255])#  20, 150, 255   255, 248, 220
    skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask = cv2.inRange(im_hsv, low_range, upper_range)
    mask = cv2.erode(mask, skinkernel, iterations=1)
    mask = cv2.dilate(mask, skinkernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (15, 15), 1)

    #plt.imshow(mask, interpolation='bicubic')
    #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    #plt.show()

    skin = cv2.bitwise_and(crop_img, crop_img, mask=mask)
    #cv2.threshold(skin, 100, 255, cv2.THRESH_BINARY)

    h, w, bpp = np.shape(skin)  # visina x sirina -> y * k

    print(h, w)

    highest_y = 0
    x_of_highest_y = -1

    for y in range(h-1, -1, -1):
        for x in range(0, w):
            if (skin[y][x] != [0, 0, 0]).all():
                highest_y = y
                x_of_highest_y = x
                print(skin[y][x])
                break
            break
        break

    print(highest_y)
    print(x_of_highest_y)

    plt.imshow(skin, interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

    return True


def testing():

    img_dir = "../../MyDataset/Levih45/Poza1"
    data_path = os.path.join(img_dir, '*g')
    files = glob.glob(data_path)
    #print(len(files))
    #print(data_path)
    print(files)
    #data = []
    for f1 in files:

        shotClockViolation(f1)

        # img = cv2.imread(f1)
        #data.append(img)
        #print(f1)


if __name__ == "__main__":

    print("Welcome to buzzer beater application :)\n\n")

    testing()

    path = "Dejan"# input("Insert relative image path for validation: ")

    contact = True#checkBallHandContact(path)
    violation = True# shotClockViolation(path)

    if contact is False:
        print("\n\nThis is VALID shot :) !")
    elif violation is True:
        print("\n\nThis shot is INVALID !!!")
    else:
        print("\n\nThis is VALID shot :) !")
