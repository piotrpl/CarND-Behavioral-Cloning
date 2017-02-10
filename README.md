#**Behavioral Cloning** 

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior or use provided dataset
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./imgs/left.jpg "Left camera"
[image2]: ./imgs/center.jpg "Center camera"
[image3]: ./imgs/right.jpg "Right camera"
[image4]: ./imgs/model.png "Architecture"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of a normalization layer and 5 convolution layers followed by the three fully connected layers (model.py lines 86-111). 
In order to introduce nonlinearity RELU activation is applied on all model layers.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py e.g. line 91). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 133-142). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 117).

####4. Appropriate training data

As a training data I used dataset provided by Udacity. 

For details about how I worked with the training data, see the next section. 

###Model Architecture and Training Strategy - Process


####1. Solution Design Approach

Given that the main task of this project was to predict steering angles the most appropriate architecture to use is convolutional neural network (CNN). 
There has been prior work done to predict vehicle steering angles from camera images, such as NVIDIA's "End to End Learning for Self-Driving Cars" (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf), and comma.ai's steering angle prediction model (https://github.com/commaai/research/blob/master/train_steering_model.py). 

In this project, I chose to use NVIDIA's model and train it using dataset provided by Udacity. The NVIDIA's model is not a complex and well documented model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
My first model was overfitting and in orer to combat this I have introduced dropout layers. This approach helped tremendously and let the vehicle drive autonomously around the track one without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 86-111) consisted of 9 layers, including:
* normalization layer
* 5 convolutional layers
* 3 fully connected layers

Data is normalized in the model using Keras lambda layer.
The next three convolution layers are 2x2 strided with 5x5 kernel, the last two convolutional layers are non-strided with 3x3 kernel. The three fully connected layers, followed the convolution ones outputing the steering angle (drive.py line 111). 


Here is a visualization of the architecture:

![alt text][image4]

####3. Training Set & Training Process

For training I used Udacity provided dataset composed of 24108 samples representing images captured from the left, center and right cameras. Here are examples of images from those trhee points of view:

![alt text][image1]
![alt text][image2]
![alt text][image3]

In the final solution 90% of the dataset was used as a training set and 10% as the validation set. When I applied recommended data split 80/20 or 70/30 my model could not drive the car through the whole track successfuly.

Data processing and image augmentation is built into the geneator, using keras fit_generator which allows to work with large amout of data. The whole dataset is not loaded into mememory but rather it is batched allowing the generator to run parallel to the model.
The following steps are taken when processing a file:

__Random selection__: The dataset contains data from three camera views: center, left and right. The view used for training is randomly chosed from the three. When using left of right view images, 0.25 is added or subracted respectively, to the steering angles.

__Minimise jitter and compensate for steering angles__: Image is randomly translated into the left or right view and compensated for the translation in the steering angles with 0.008 per pixel of translation. Then, the region of iterested is cropped out.

__Random flip__: Images are randomly flipped together with the change of the sign (+/-) of the steering agles in other to balance left and right images.

__Brightness augmentation__: By converting image to HSV channel and randomly scaling the V channel different light conditions e.g. shadow are simulated are simulated.

During training I have tried using various number of epochs (from 10 to 30) but given the results I have settled on 10 as the ideal number. Similary I have tried applying different values for the Droupout (from 0.1 to 0.5) but the value of 0.2 provided best results.
I tried applygling l2 regularization on the first and other layers with different values (from 0.001 to 0.01) but again for this particualr case not having regularization allowed my model to drive the car through the whole track.

I used an adam optimizer so that manually training the learning rate wasn't necessary.

###Fututre considerations
* Implement the same solution using comma.ai model. It would be interesting to compare the two models.
* Use ELU's instead of the ReLU's for activation. It would be beneficial to have comparison data backed by this use case.
* Try using l2 regularization. I believe that this could further smoothen out car behaviour.
* Manage speed according to the steering angle. I expect this could help minimise car jittering.
* Use smaller range of values than [-0.25, 0.25] when selecting views from side cameras. I expect this could further minimise car jittering.
