import cv2
import numpy as np


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

    print(imgPath)

    return False


if __name__ == "__main__":
    print("Welcome to buzzer beater application :)\n\n")

    path = input("Insert relative image path for validation: ")

    contact = checkBallHandContact(path)
    tableSemaphore = checkShotTime(path)

    if contact is False or (contact is True and tableSemaphore is True):
        print("\n\nThis is VALID shot !")
    else:
        print("\n\nThis shot is INVALID !!!")
