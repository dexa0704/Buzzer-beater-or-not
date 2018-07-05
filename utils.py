import glob
import os

# Slobodno bacanje !

bacanje1_crop_img = [880, 950, 1020, 1110]  # - Poza1
bacanje2_crop_img = [1000, 1080, 1128, 1280]  # - ovo je sa 5:57 tura slika - Poza2
bacanje3_crop_img = [690, 830, 1108, 1225]  # - poza 3
bacanje4_crop_img = [870, 980, 1228, 1420]  # - ovo je za semaphoe four.. 3:09 ta tura slika - Poza4 - jedina mix poza
bacanje5_crop_img = [693, 760, 1170, 1300]  # - poza 5

bacanje_global = [870, 950, 1128, 1280]

# Desnih 45

desnih45_1_crop_img = [580, 680, 1910, 1940]  # - Poza1
desnih45_2_crop_img = [690, 760, 1935, 2050]  # - Poza2
desnih45_3_crop_img = [650, 710, 1780, 1910]  # - Poza3
desnih45_4_crop_img = [570, 690, 1880, 2030]  # - Poza4
desnih45_5_crop_img = [690, 810, 1900, 2080]  # - Poza5

desnih45_global = [570, 810, 1880, 2080]

# Levih 45

levih45_1_crop_img = [975, 1085, 910, 1100]  # - Poza1
levih45_2_crop_img = [905, 1090, 910, 1205]  # - Poza2
levih45_3_crop_img = [660, 795, 505, 680]  # - Poza3
levih45_4_crop_img = [610, 660, 305, 400]  # - Poza4
levih45_5_crop_img = [750, 770, 360, 490]  # - Poza5
levih45_6_crop_img = [660, 800, 470, 645]  # - Poza6
levih45_7_crop_img = [570, 650, 175, 270]  # - Poza7

levih45_global = [570, 770, 300, 680]


def getDistance(pixel, refcolor):

    return abs((pixel[0]-refcolor[0]) + (pixel[1]-refcolor[1]) + (pixel[2]-refcolor[2]))


def testingShotclockByPose():

    '''
    Helper function for testing

    :return: nothing ...
    '''

    img_dir = "../../MyDataset/Levih45/Poza2-Mesano"
    data_path = os.path.join(img_dir, '*g')
    files = glob.glob(data_path)
    # print(len(files))
    # print(data_path)
    print(files)
    # data = []
    for f1 in files:

        pass
        # shotClockViolation(f1)

        # img = cv2.imread(f1)
        # data.append(img)
        # print(f1)


def performCompatibleCrop(dirs):

    positions = dirs.split('\\')[-2]
    pose = dirs.split('\\')[-1].split('-')[0]

    print(positions)
    print(pose)

    if positions == "Bacanje":

        '''

        if pose == "Poza1":

            return bacanje1_crop_img

        if pose == "Poza2":

            return bacanje2_crop_img

        if pose == "Poza3":

            return bacanje3_crop_img

        if pose == "Poza4":

            return bacanje4_crop_img

        if pose == "Poza5":

            return bacanje5_crop_img

        '''

        return bacanje_global

    elif positions == "Desnih45":

        '''

        if pose == "Poza1":

            return desnih45_1_crop_img

        if pose == "Poza2":

            return desnih45_2_crop_img

        if pose == "Poza3":

            return desnih45_3_crop_img

        if pose == "Poza4":

            return desnih45_4_crop_img

        if pose == "Poza5":

            return desnih45_5_crop_img
        '''

        return desnih45_global

    elif positions == "Levih45":

        '''

        if pose == "Poza1":

            return levih45_1_crop_img

        if pose == "Poza2":

            return levih45_2_crop_img

        if pose == "Poza3":

            return levih45_3_crop_img

        if pose == "Poza4":

            return levih45_4_crop_img

        if pose == "Poza5":

            return levih45_5_crop_img

        if pose == "Poza6":
            return levih45_6_crop_img

        if pose == "Poza7":
            return levih45_7_crop_img

        '''
        return levih45_global
