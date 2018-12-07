# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/stop.png "Traffic Sign 1"
[image5]: ./examples/no_entry.jpg "Traffic Sign 2"
[image6]: ./examples/30.jpeg "Traffic Sign 3"
[image7]: ./examples/70.jpg "Traffic Sign 4"
[image8]: ./examples/turn_right_ahead.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


### Data Set Summary & Exploration

##### 1. Basic summary of the data set:

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32, 32, 3
* The number of unique classes/labels in the data set is 43

##### 2. Visualization of the dataset:

Here is an exploratory visualization of the data set. It is a bar chart showing the data distribution in terms of training, validation and test data.

![alt text][image1]

### Design and Test a Model Architecture

#####1. I've peprocessed the images first by converting them to relative luminance (as suggested in Stanford CNN course (CS231n) and then normalized them.


#####2. My final model consisted of the following layers:

| Layer         		|     Description	        | 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   | 
| Convolution 5x5       | 1x1 stride, valid padding, outputs 28x28x64 |
| RELU					|						|
| Max pooling	      	| 2x2 stride,  outputs 14x14x64 |
| Dropout               | rate of 0.5	             |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32|
| RELU		            |              	|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32   |
| Flatten				| output = 800		|
| Dropout               | used 0.5 keep_prob while training             |
| Fully Connected       | input = 800 output = 400                      |
| RELU		            |              	|
| Fully Connected       | input = 400 output = 200                      |
| RELU		            |              	|
| Dropout               | used 0.5 keep_prob while training             |
| Fully Connected       | input = 200 output = 43 (n_classes)           |


#####3. I've initally used %20 of the training data to find a model with enough layers to overfit. I should also mention that I've used Adam optimizer during the training since Adam is generally regarded as being fairly robust to the choice of hyper parameters. This was suggested by various resources including Stanford CS231n CNN course. 

##### 4. I've used the mean and stddev values that were being used in the course material: mu = 0, sigma = 0.1. I've also used batch size of 128. I've found the 'optimal' learning rate and epoch size empirically. I've initially used 0.001 for learning rate and epoch size of 25, but this model was reaching to the maximum rate fairly quickly (around pass 10) and not improving further since the learning rate resolution was too big. Then I've tried 0.00001 with an EPOCH size of 100, but this was under fitting since both the training and validation sets had low accuracy. Then after a few tries I've used learning rate of 0.00038 and EPOCH size of 65 which seemed to be optimal as far as I could conclude.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.957 
* test set accuracy of 0.999

The first architecture was I had chosen was LeNet since I had already created one from the previous quiz, and I thought this would be a good starting point to experiment.
I wasn't getting good enough accuracy with this architecture, so I've added more layers to experiment. This was I was able to improve the accuracy significantly. Then I've added dropouts to prevent overfitting and this helped improving the accuracy.
I've tuned the learning rate and epoch 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		|  33, Turn right ahead  | 
| No entry     			|  13, Yield	|
| Speed limit 30			| 	25, Road work	|
| Speed limit 70		    | 	4, Speed limit 70|
| Right turn ahead		|  2, Speed limit 50    |


The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%. The accuracy on the on the random data set is realy poor. There are number of factors contributing to this poor performance. For examples these images don't have a consistent aspect ratio, when scaled down they will be severely distorted. Some of the signs are also shown from an angle which also makes it harder to recognize. We would probably need a much larger training data set to be able to improve the accuracy.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


