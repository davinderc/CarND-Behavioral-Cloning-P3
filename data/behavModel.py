import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Read in the data, starting with the CSV that provides image paths and steering angles
lines = []
with open('../very_short/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
b = 1
c_images = []
measurements = []
for line in lines:
    source_path_center = line[0] # Image paths
    source_path_left = line[1]
    source_path_right = line[2]

    center_filename = source_path_center.split('/')[-1] # Image filenames
    left_filename = source_path_left.split('/')[-1]
    right_filename = source_path_right.split('/')[-1]

    center_path = '../very_short/IMG/' + center_filename # Updated pathnames
    left_path = '../very_short/IMG/' + left_filename
    right_path = '../very_short/IMG/' + right_filename

    center_image = cv2.imread(center_path) # Read in images
    left_image = cv2.imread(left_path)
    right_image = cv2.imread(right_image)

    if center_image is not None: # Create dataset using images/steering angles
        c_images.append(center_image)
        measurement = float(line[3])
        measurements.append(measurement)
    if left_image is not None:
        c_images.append(left_image)
        measurement = float(line[3]) + 0.2
        measurements.append(measurement)
    if right_image is not None:
        c_images.append(right_image)
        measurement = float(line[3]) - 0.2
        measurements.append(measurement)

# Numpy arrays for training data
X_train = np.array(c_images)
y_train = np.array(measurements)

# Basic Keras model

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,20),(0,0))))
model.add(Convolution2D(6,5,5, border_mode='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), border_mode='valid'))
model.add(Convolution2D(15,5,5, border_mode='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), border_mode='valid'))
model.add(Dense(100))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=30)

model.save('model.h5')
