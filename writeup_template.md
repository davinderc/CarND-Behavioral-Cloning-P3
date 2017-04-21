# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image3]: ./center_2017_04_20_11_26_54_919.jpg "Center Image"
[image4]: ./left_2017_04_20_11_26_54_919.jpg "Left Recovery Image"
[image5]: ./right_2017_04_20_11_26_54_919.jpg "Right Recovery Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
#### Files Submitted & Code Quality

###### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md and writeup_report.pdf summarizing the results

###### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

###### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

#### Model Architecture and Training Strategy

###### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on the LeNet architecture. It starts with a layer that normalizes the input to between -0.5 and 0.5. Then the input images are cropped to remove 70 pixels from the top and 20 from the bottom. (model.py lines 55-57)

Next, a convolutional layer with 5x5 filter sizes and a depth of 6 is used, with ReLU activation for nonlinearity. A pooling layer with 3x3 kernel comes next, followed by another convolutional layer with 5x5 filters and a depth of 15, with ReLU activation and another pooling layer after that (same 3x3 kernel).(model.py lines 57-61)

Here I diverged from the LeNet architecture, by introducing a single fully connected layer of size 100, followed by a dropout set at 0.65. Finally the output is flattened and followed by an output layer of size 1, to produce the steering angle. (model.py lines 62-65)

###### 2. Attempts to reduce overfitting in the model

The model contains a single dropout layer to reduce overfitting (model.py lines 63).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 67-68). The validation split was 20%. However, I felt that due to the way I chose my dataset, validation was somewhat useless. This will be further explained below in the data collection section.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. The model was tested at both 15mph and 20mph, the latter of which was somewhat less stable, but still managed to remain on the track.

###### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 67).

###### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used left/right camera images as to make sure the model knew how to recover from drifting too far to the left/right, as well as center camera images to train the model to drive down the middle of the track as much as possible. I did not use flipped images or reverse track driving images, although these would help the model to generalize further and drive on more diverse tracks.

For details about how I created the training data, see the next section.

#### Model Architecture and Training Strategy

###### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to create convolutional layers based on the LeNet architecture, with kernel sizes that seemed appropriate for the sizes of the track edges.

Initially, I used a single fully connected layer just to see some form of autonomous driving in action, and confirm that everything was working properly. This worked, even if the model appeared to be drunk, although at some point in the track the car went off of the road and no amount of adjusting or new data seemed to correct that.

My next step was to use a convolution neural network model similar to the LeNet architecture as suggested in the lessons. I thought this model might be appropriate because the edges of the track were simple lines and this architecture worked very well on classifying more complex traffic signs. Since the model only needed to output a steering angle, three convolutional layers were followed by a single fully connected layer and a dropout layer.

However, unintentionally, the flatten layer was placed after the dropout layer, when it really should have gone immediately after the last convolutional layer. However, the model is able to drive the track very well and with tweaking can probably do an even more stable job, so I decided to keep this model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean absolute error on the training set and a slightly higher mean absolute error on the validation set. This implied that the model might be overfitting and to combat this I used a single dropout layer after the fully connected layer, to make sure that no single pathway would learn the dataset perfectly.

The final layer was a single output for the steering angle.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track and I added individual images and steering angles of the car in those locations to improve the driving behavior in these cases. I chose steering angles by hand based on the collected images and tweaked the steering angles as necessary to make sure the model could drive the car appropriately in these situations.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. The vehicle does seem to weave back and forth on the road, which may be due to inaccuracies in choosing steering angles, as well as imperfections in the parameters of the model.

###### 2. Final Model Architecture

The final model architecture (model.py lines 55-65) consisted of a convolution neural network with the following layers and layer sizes (the layer sizes were chosen somewhat arbitrarily, and upon calculating all the sizes below, I now realize that the filter sizes and layer sizes could have been chosen more appropriately, especially since there is a larger layer closer to the output. However, considering that I adjusted the dataset meticulously to make the model drive, I decided to keep the model. Further improvements to this model would involve resizing the layers appropriately):

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input (after cropping)           | 70x320x3 RGB image   	|
| Convolution 5x5     	| 5x5 stride, valid padding, outputs 14x64x6 	|
| ReLU					  |												|
| Max pooling	    | 3x3 stride, same padding, outputs 5x22x6 				|
| Convolution 5x5	| 5x5 stride, valid padding, outputs 1x4x15      	  |
| ReLU            |                       |
| Max pooling     | 3x3 stride, same padding, outputs 1x3x15          |
| Fully connected	| 45 inputs            |
| ReLU            |                       |
| Dropout         | Keep probability 0.65  |
| Fully connected | 100 inputs and final output of 1 steering angle           |

###### 3. Creation of the Training Set & Training Process

I decided to collect data in a less conventional manner. Instead of learning to drive the simulator with a mouse and then collecting thousands of images for training, to later use on an AWS GPU enabled instance, I was inspired by a classmate to curate my own dataset.

I chose to collect a few images at various places on the track, and decide on an appropriate steering angle for that location by hand. The left/right camera images were also used, with fixed offset values for the steering angles.

Before starting, I imagined, from looking at the simulator graphics, that the model would decide the steering angle from detecting the edges of the track, which meant that a simple model would probably suffice for an approximate model. However, no amount of tweaking of the dataset or the parameters seemed to stop the car from crashing at a given point on the track and I decided to use a more complex network.

Considering that the relevant features were simply the edges of the road and their shapes and angles, a select few images should be enough to decide on steering angles.

Initially only 35 center cameras were used, but after adding in left/right camera images and additional examples for problem areas, the total came to 228 images. This ensured that all types of road edges were considered and that the dataset was not horribly imbalanced in regards to the different road edge appearances.

The right/left images were used to teach the model to recover from being too far to the side of the track, which proved to be useful when the car drifted off to one side. Below are some examples of the images, starting with center, left and then right images.

![alt text][image3]
Center camera image

![alt text][image4]
Left camera image

![alt text][image5]
Right camera image

After the collection process, I had 228 data points. I then preprocessed this data by normalizing the images and cropping them down to a size that only showed the road.

During training and adjustment of the dataset, it was found that cropping too much off of the dataset would cause the model to have difficulties in differentiating between different sections of the track. This would result in decreasing performance in certain parts of the track as other parts showed better performance after new data was trained on.

Once cropping was done less aggressively, the model was able to improve on individual sections of the track, despite similarities in road edges.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting, although I'm not sure how reliable it would be with this small dataset. The ideal number of epochs was 7 as evidenced by the nonincreasing validation loss.
