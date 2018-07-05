import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
from utils import *

rootDataSet = "../../MyDataset"
supported_positions = [ "Bacanje", "Desnih45", "Levih45"]


def shotClockViolation(imgPath, crop):

    '''

    Function for checking time on table semaphore. If zero on table, shot isn't valid. Checking specific circle on the
    semaphore. If that circle exist clock is on zero, if circle doesn't exist time still counting ...


    :param imgPath: relative path for image for checking time on table
    :param crop: dimension for cropping image bacause of easier detection shot clock,
        list with 4 elements: y from, y to, x from, x to
    :return: boolen represent presents of violation, true -> there is a violation, false -> no violation

    '''

    im = cv2.imread(imgPath)

    crop_img = im[crop[0]:crop[1], crop[2]:crop[3]]

    # plt.imshow(crop_img, interpolation='bicubic')
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # plt.show()

    bilateral_filtered_image = cv2.bilateralFilter(crop_img, 5, 175, 175)

    gray = cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    edge_detected_image = cv2.Canny(gray, 75, 200)
    # cv2.imshow('Edge', edge_detected_image)
    # cv2.waitKey(0)

    _, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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

    im2 = cv2.imread(imgPath)

    # ball counture part

    im = im2[50:, 950:]

    bilateral_filtered_image = cv2.bilateralFilter(im, 5, 175, 175)

    gray = cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    edge_detected_image = cv2.Canny(gray, 75, 200)

    # cv2.imshow('Edge', edge_detected_image)
    # cv2.waitKey(0)

    _, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rows = gray.shape[0]

    circles = cv2.HoughCircles(edge_detected_image, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=30,
                               minRadius=50, maxRadius=100)

    # print(circles)

    if circles is None:

        print("Contact not detected ...")
        return False

    '''
    cimg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        #cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    '''
    # plt.imshow(cimg, interpolation='bicubic')
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # plt.show()

    # cv2.imshow('detected circles', cimg)
    # cv2.waitKey(0)

    # skin part

    crop_img = im[950:, ]

    im_hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    low_range = np.array([0, 10, 60])# 0, 10, 60   128,0,0
    upper_range = np.array([20, 150, 255])#  20, 150, 255   255, 248, 220
    skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask = cv2.inRange(im_hsv, low_range, upper_range)
    mask = cv2.erode(mask, skinkernel, iterations=1)
    mask = cv2.dilate(mask, skinkernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (15, 15), 1)

    # plt.imshow(mask, interpolation='bicubic')
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # plt.show()

    skin = cv2.bitwise_and(crop_img, crop_img, mask=mask)
    cv2.threshold(skin, 100, 255, cv2.THRESH_BINARY)

    # plt.imshow(skin, interpolation='bicubic')
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # plt.show()

    h, w, bpp = np.shape(skin)  # visina x sirina -> y * x

    print(h, w)

    highest_y = 0
    x_of_highest_y = 0

    for y in range(h-1, -1, -1):
        for x in range(0, w):
            if (skin[y][x] != [0, 0, 0]).all():
                highest_y = y
                x_of_highest_y = x
                print(skin[y][x])
                break
            break
        break

    print(circles[0][0])
    print(skin[highest_y][x_of_highest_y])

    # distance.euclidean(skin[0][0], circles[0])

    distance = getDistance(circles[0][0], skin[highest_y][x_of_highest_y])
    print(distance)

    if distance < 1999:
        print("Detect contact !")
        return True
    else:
        print("Contact not detected ...")
        return False


def checkOneShot(path, crop):
    '''
    Function calling two main methods for image processing and that results apply on basketball logic

    :param path: path to the image for checking
    :param crop: crop sizes for shot clock violation
    :return: boolean representation of validation shot. True -> valid shot, False -> invalid...
    '''

    contact = checkBallHandContact(path)
    violation = shotClockViolation(path, crop)

    if contact is False:
        print("\n\nThis is VALID shot :) !")
        return True

    elif violation is True:
        print("\n\nThis shot is INVALID !!!")
        return False

    else:
        print("\n\nThis is VALID shot :) !")
        return True


def testProgramOnAllData(root):

    '''
    Main function for testing application with all data.

    :param root: path to root folder with data set and all images
    :return: percentage of success
    '''

    number_images = 1
    number_correct_solved = 0
    current_image_validity = False

    for pose in supported_positions:

        pose_path = rootDataSet+"/"+pose
        directories = filter(os.path.isdir, [os.path.join(os.path.realpath(pose_path), p) for p in os.listdir(pose_path+'/')])

        print(pose_path)

        for dirs in directories:

            # complet_path = pose_path+"/"+dirs
            data_path = os.path.join(dirs, '*g')
            files = glob.glob(data_path)

            print(dirs)

            folder_sign = dirs.split('-')[1]
            print(folder_sign)

            if folder_sign == "Regularno":

                current_image_validity = True

            elif folder_sign == "Neregularno":

                current_image_validity = False

            for image in files:

                print(image)

                image_sign = image.split('-')
                print(image_sign)

                if len(image_sign) > 2:

                    if image_sign[2].split('.')[0] == "regularno":

                        current_image_validity = True

                    elif image_sign[2].split('.')[0] == "neregularno":

                        current_image_validity = False

                number_images += 1

                crop_coordinates = performCompatibleCrop(dirs)
                print(crop_coordinates)

                validation = checkOneShot(image, crop_coordinates)

                if validation == current_image_validity:

                    number_correct_solved += 1

    print("Correctly solved: "+str(number_correct_solved))
    print("Whole data set contains "+str(number_images)+" images.")
    return (number_correct_solved / number_images) * 100
