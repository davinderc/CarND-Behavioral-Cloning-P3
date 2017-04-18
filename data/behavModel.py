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
    source_path_center = line[0] # Center image paths
    center_filename = source_path_center.split('/')[-1]
    center_path = '../very_short/IMG/' + center_filename
    center_image = cv2.imread(center_path) # Read in center images
    if b == 1:
        #print(center_image)
        b = 0
    if center_image is not None:
        c_images.append(center_image)
        measurement = float(line[3])
        measurements.append(measurement)
#########
# Must check dataset, since files are missing!!! This causes array size errors!!!
#########

   # print(np.array(c_images).shape)

#print(len(measurements))
# Numpy arrays for training data
X_train = np.array(c_images)
y_train = np.array(measurements)

#print(len(c_images[0]))
#print(X_train.shape)
#test = X_train.reshape((1,102))
#print(X_train.shape)
#print(y_train.shape)

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
