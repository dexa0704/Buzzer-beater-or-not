from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 16, 12


#####################################################################################################################
# Helper functions
#
# List of functions:
#
#
#
#####################################################################################################################


def load_image(path):

    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):

    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin


def invert(image):

    return 255-image


def display_image(image, color=False):

    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')


def dilate(image):

    kernel = np.ones((3, 3))
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):

    kernel = np.ones((3, 3))
    return cv2.erode(image, kernel, iterations=1)


#####################################################################################################################
# Region of interest
#
# List of functions:
#
#
#
#####################################################################################################################


def resize_region(region):

    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)


def select_roi(image_orig, image_bin):

    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        if area > 100 and h < 100 and h > 15 and w > 20:
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = sorted_regions = [region[0] for region in regions_array]

    return image_orig, sorted_regions


def scale_to_range(image):

    return image/255


def matrix_to_vector(image):

    return image.flatten()


def prepare_for_ann(regions):

    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))

    return ready_for_ann


def convert_output(alphabet):

    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)

    return np.array(nn_outputs)


#####################################################################################################################
# Neural network
#
# List of functions:
#
#
#
#####################################################################################################################


def create_ann():

    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(10, activation='sigmoid'))

    return ann


def train_ann(ann, x_train, y_train):

    x_train = np.array(x_train, np.float32)  # dati ulazi
    y_train = np.array(y_train, np.float32)  # zeljeni izlazi za date ulaze

    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    ann.fit(x_train, y_train, epochs=2000, batch_size=1, verbose=0, shuffle=False)

    return ann


def winner(output):

    return max(enumerate(output), key=lambda x: x[1])[0]


def display_result(outputs, alphabet):

    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])

    return result
