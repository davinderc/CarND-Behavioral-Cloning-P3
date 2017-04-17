import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense

# Read in the data, starting with the CSV that provides image paths and steering angles
lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
b=1
c_images = []
measurements = []
for line in lines:
    source_path_center = line[0] # Center image paths
    center_filename = source_path_center.split('/')[-1]
    center_path = '../data/IMG/' + center_filename
    center_image = cv2.imread(center_path) # Read in center images
    if b == 1:
        print(center_image)
        b = 0
    if center_image is not None:
        c_images.append(center_image)
#########
# Must check dataset, since files are missing!!! This causes array size errors!!!
#########

    print(np.array(c_images).shape)
    measurement = float(line[3])
    measurements.append(measurement)

print(len(measurements))
# Numpy arrays for training data
X_train = np.array(c_images)
y_train = np.array(measurements)

print(len(c_images[0]))
print(X_train.shape)
test = X_train.reshape((1,102))
print(X_train.shape)
print(y_train.shape)

# Basic Keras model

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

model.save('model.h5')
