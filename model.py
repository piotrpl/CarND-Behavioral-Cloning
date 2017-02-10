import os
import argparse
import json
import cv2
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D
from keras.preprocessing.image import load_img, img_to_array

def trans_image(image, steer, trans_range):
    rows, cols, chan = image.shape

    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * .4

    Trans_M = np.float32([[1, 0, tr_x],[0, 1, 0]])
    image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))

    return image_tr, steer_ang

def adjust_brightness(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    image[:, :, 2] = image[:,:,2] * random_bright
    return image

def adjust_size(img):
    img = img[60:140,40:280]
    return cv2.resize(img, (number_of_columns, number_of_rows))

def process_line_data(line_data):
    view = np.random.randint(3)
    if (view == 0):
        img_path = line_data['left'][0].strip()
        shift_steering = .25
    if (view == 1):
        img_path = line_data['center'][0].strip()
        shift_steering = 0.
    if (view == 2):
        img_path = line_data['right'][0].strip()
        shift_steering = -.25

    steering = line_data['steering'][0] + shift_steering
    image = load_img(data_path + img_path)

    image, steering = trans_image(img_to_array(image), steering, 50)
    image = adjust_brightness(img_to_array(image))
    image = adjust_size(image)

    image = np.array(image)
    ind_flip = np.random.randint(2)
    if ind_flip==0:
        image = cv2.flip(image, 1)
        steering = -steering

    return image, steering

def batch_generator(data, batch_size = 256):
    batch_images = np.zeros((batch_size, number_of_rows, number_of_columns, 3))
    batch_steering = np.zeros(batch_size)
    while 1:
        for batch_index in range(batch_size):
            line_index = np.random.randint(len(data))
            line_data = data.iloc[[line_index]].reset_index()

            image, steering = process_line_data(line_data)

            image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
            steering = np.array([[steering]])

            batch_images[batch_index] = image
            batch_steering[batch_index] = steering

        yield batch_images, batch_steering

def get_model():

    model = Sequential()

    # Normalization layer
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(number_of_rows, number_of_columns, 3), name='Normalization'))

    # Convolution layers
    # strided
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu', W_regularizer = l2(0.001), name='Conv1'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu', name='Conv2'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu', name='Conv3'))
    model.add(Dropout(0.2))
    # non-strided
    model.add(Convolution2D(64, 3, 3, activation='relu', name='Conv4'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='Conv5'))
    model.add(Dropout(0.2))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(100, activation='relu', name='FC2'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu', name='FC3'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu', name='FC4'))

    # Output layer
    model.add(Dense(1, name='output'))

    # print summary of the model
    model.summary()

    opt = Adam(lr=0.0001)
    model.compile(optimizer = opt, loss='mse', metrics = [])

    return model;


data_path = './data/'
log_path = './data/driving_log.csv'

log_file_columns = ['center', 'left', 'right' , 'steering', 'throttle', 'brake', 'speed']

driving_log = pd.read_csv(log_path, skiprows=[0], names = log_file_columns)

number_of_rows = 66
number_of_columns = 200


training_data = driving_log.sample(frac=0.9)
validation_data = driving_log.sample(frac=0.1)
training_data_generator = batch_generator(training_data)
validation_data_generator = batch_generator(validation_data)

model = get_model()

model.fit_generator(training_data_generator, samples_per_epoch = 20480, nb_epoch = 10,
                    validation_data = validation_data_generator, nb_val_samples = len(validation_data.index),
                    verbose = 1)

model_json = model.to_json()
with open('model.json', "w") as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')
