Object Pose Tracking in 3D Space by Using Two Webcams
===

Do-Men Su, Bo-Hsun Chen

## Introduction
 

[Project Proposal](./proposal.pdf)

[Project Presentation](./presentation.pdf)

## Motivation
Object pose estimation in real time is an important technique in computer vision, and it can be applied in many fields. Such helpful information can facilitate many attractive ongoing researches, such as a robotic arm needs to see and judge an object precisely, so it can properly catch and grasp it. And, a self-driving cars should estimate poses of pedestrians, other cars, or barriers on the street, so it can avoid them and safely drive on the road. 

To estimate the object pose in 3D space, we need more than an RGB picture. However, many current research used RGB-D or point-cloud cameras, which are expensive and hard to access for ordinary people. So in this project, we try to build a custom-made RGB-D vision system composed of two easily-accessible commercial 2D webcams. And then, we try to estimate the object pose based on the images from our RGB-D vision system, so we can achieve a cheaper soluion. 



## Approaches

To conduct our experiment, we first used OpenCV to take two images simultaneously from the webcams. After having multiple paired stereo images, OpenCV functions were used to generate the corresponding disparity maps. And finally, we trained the neural network models with the RGB images and the disparity maps.

Besides training the neural network model, we additionally recorded a stereo video as the testing data. In the video, we move our object around with different poses. Therefore, we utilized a histogram-based tracker and the neural network model to show the pose and position of the object in the video.

### Experiment Setup

Our experiment setup is shown in figure 1. We tied two Logitech C310 webcams together and put them on a camera tripod. The object for our experiment was a magic cube, which was put on a black carton box.

The background was covered with a black piece of paper originally. This approach made it easier to remove the background since we only needed to erase the black parts of the images. However, when building the dataset, we found that such background caused the magic cube overexposed. Therefore, the setup shown in figure 1 used a white background instead.

![](https://i.imgur.com/U2plG6q.png)  
Figure 1: Experiment setup


### Building Datasets

To build the training data, we manually rotated the magic cube on the carton box through both Y and Z axes. As shown in figure 2, X axis pointed out of the paper, Y axis pointed to the right, and Z axis pointed upwards. The animation of figure 3 shows that we rotated the magic cube through Z axis for a round; lift it up (rotation through Y axis) and rotated it through Z axis for another round again. As a result, we took 197 paired pictures in total as our dataset.

![](https://i.imgur.com/HBIWZwH.png)  
Figure 2: The three axes used to build the dataset

![](https://i.imgur.com/Tp2c9ii.gif)  
Figure 3: The animation of rotating the magic cube through Z axis


### Building Disparity and Depth Maps with Two Webcams

To generate the RGB-D images from the paired RGB images, we needed some algorithms to generate the depth maps. Therefore, OpenCV functions are utilized to process the paired images. To generate the depth maps in OpenCV, we needed the following steps. The first is to generate the disparity maps. And, secondly, we needed to calibrate the cameras in order to compute the depth maps from the disparity maps. To calibrate our webcams, we originally used `checkerboardPattern.pdf` from Matlab.

As told in the class, the relationship of the disparity and the depth is based on the focal length $$f$$ and the horizontal baseline $$b$$ as shown in the following formula: $$z = \frac{bf}{d}$$, where $$z$$ is the depth and $$d$$ is the disparity. As we can see from our experiment setup, the baseline and the focal length were static throughout the experiment, so the depth was inversely proportional to the disparity. Since the follow-up usage of the depth maps was the neural network model, it is not meaningful to compute such linear transformation; hence, for simplicity, we took the disparity maps as the input of the neural network model instead of the depth maps.

The algorithm chosen to compute the disparity is Semi-Global Block Matching (SGBM). Using a block-based approach, the algorithm estimates the disparity from the left and the right images. This algorithm is implemented in OpenCV as `cv2.StereoSGBM_create`. To achieve a satisfactory result, we needed to tune a few parameters in the function. To help ourselves tunning the parameter, we built an instant GUI previewer (figure 4) of the disparity map, and listing 1 shows parameters chosen in our experiment.

![](https://i.imgur.com/7MMhzjb.png)  
Figure 4: The GUI disparity preview tool


```
numDisparities: 64
blockSize: 5
uniquenessRatio: 0
disp12MaxDiff: 50
speckleWindowSize: 0
speckleRange: 0
P1: 8 * blockSize ** 2
P2: 32* blockSize ** 2
```
Listing 1: The parameters used for `StereoSGBM_create`

- `numDisparities` controls the maximum number of pixels difference of the same point on the two images. If `numDisparities` is too large, the size of unused pixels would be large. The unused pixels are shown on the left side of the disparity map in figure 4. However, if `numDisparities` is too small, the algorithm may fail to match the corresponding pixel because their difference is larger than `numDisparities`.
- `blockSize` is the block size used to compare the two images. Smaller `blockSize` makes the result more accurate but may fail in some pixels.
- `uniquenessRatio` is the threshold that measures the reliableness of disparity. The unreliable pixels would be marked as `-16`, which may cause additional problems for our neural network training process, so we would rather get an unreliable value instead.
- `disp12MaxDiff` is the maximum allowed difference in two images. If the value is too small, the algorithm would fail on some pixels.
- The rest of the variables control the smoothness of the comparison and we set them to a suggested value in the OpenCV document.


### CNN Structure

We use convolutional neural network (CNN) model to infer the relationship between input images and the object orientation. There are multiple ways to represent object orientations, including rotation matrix, Euler angle, and quaternion. Since rotation matrix has 9 paramters with many constraints (orthonormal matrix), and Euler angles has the singularity problem, we use the quaternions, which use four numbers to represent the object pose, with Euclidean norm equaling to 1 as the only constraint.

For our dataset, we only collected 198 images and we did not rotate the object along the X-axis of the global frame. But, since we have aligned the camera frame to the global frame, the effect of rotating the image (along the X-axis of the camera frame) is almost equivalent to rotating the object along the X-axis of the global frame. So, we augment our dataset by rotating each original image in the dataset by specific angles (20, 40, ..., 340 deg), and then we can get 18 times larger data set.

Referencing the PoseCNN structure in [1], we build our CNN structure as the following figure.

![Figure X. The CNN structure](https://i.imgur.com/EnbZ499.png)

The input data are normalized images, and the output data are their normalized quaternion values. There are three convolutional 2D layers in this structure, which can extract the features from images. And each convolutional layer is followed by the batch normalization to assist in training and a max-pooling 2D layer. Then, two fully connected layers are concatenated to infer the relationships between the features and the quaternion values. Lastly, a layer of 4 sigmoid activation functions is set as the output to generate the quaternion. The loss function is set as

![log( sum( ( y_i - y_hat_i )^2, i = 1 to N ) )](https://i.imgur.com/vnaCtZ0.png)

, where logarithm can magnify and distinguish small differences between loss values when the loss values become very small during training progress. We use Adam as the CNN training optimizer, and we only save the model parameters with the lowest training loss during training progress.

To compare the performance of our idea, we input three different types of images to create three different types of models. The first one is inputting only the RGB image with 3 channels from the left camera, the second one is inputting the RGB image from the left camera as well as the disparity map with total 4 channels, and the last one is inputting two RGB images from the two cameras with total 6 channels. We compare the training progress of the three different models. The results and the table of final training and validation losses are shown as follows.

![Figure 2. training progresses of three models](https://i.imgur.com/v8T5rq6.png)

|                     | RGB      | RGB-D    | RGB-RGB | 
| ------------------- | -------- | -------- | ------- |
| final training loss | -4.5035  | -4.5839  | **-4.6329** |
|  final valid loss   | -4.7390  | -4.7038  | **-4.9640** |

The training progresses show that all the three models of different input image types are trainable since their training losses decrease as the epoch numbers grow. Besides from the table, the CNN model of two-RGB-image-input has the lowest final training and validation loss, indicating it may perform best. On the other hand, the model of RGBD-input has the highest validation loss, indicating that it may not be better than the model of only inputting an RGB image.

Since the model of two-RGB-image-input has the best indexes in this stage, we apply it on some randomly chosen validation data, and compare the plotted arrows of predicted axes and original label axes by visualization to judge its performance on predicting object pose. The following figure shows the prediction results.

![prediction results of model with two-image-input](https://i.imgur.com/6ELVoxS.jpg)

The images above are using the left images from the two paired images of each validation data for visualization. And, the figure shows that for some images, like (b), (d), and (e), the model can predict the object pose very well and make the generated arrows align to the original ones. For some images, like (a) and (i), they have some degree biases between the corresponding arrows, indicating not predicting well. While for some images, like (f), it predicts totally wrong pose of the object.

So, the model could be not stable.

## Test on Video: Result
Finally, we compare the three models by applying them on a series of recorded images. In the beginning of the video, the cube was placed on the black podium with its X, Y, and Z-axis of its body-fixed frame right along the directions pointing out (X-axis of the global frame), left (Y-axis of the global frame), and up (Z-axis of the global frame) of the images. Then, we took the cube out and arbitrarily rotated and moved it in the scene for a duration of time. And then, the models should predict the object poses by plotting the colored arrows (red arrow for X-axis, green arrow for Y-axis, and blue arrow for Z-axis) in correct directions. The following videos show the testing results.

model of one-RGB-image-input:
https://drive.google.com/file/d/1uZhYTz5cEHzECFc94DMwRKBaaMD4e4ii/view?usp=sharing

model of RGBD-input:
https://drive.google.com/file/d/1lsD0sH9KeJC8ZzKpUTkOqrb_jngJZEbD/view?usp=sharing

model of two-RGB-image-input:
https://drive.google.com/file/d/1p9Jdv8JFcTpLyByrP9wgYnTrIoBMBlzT/view?usp=sharing

These videos show that the model of one-RGB-image-input performs best. Most of its X-axis arrows were steadily attached perpendicular to the cube’s front face with its Y and Z-axes arrows lying on the front face the cube and pointing to the correct directions, indicating the model predicts the object pose well. While for the other two models, the arrows vibrate severely, and sometimes randomly point to totally wrong directions.

### Discussion

We think there are some possible reasons which cause the model of one-RGB-image-input is better and more stable than the models of RGBD-input and two-RGB-image-input. Firstly, we trained our model under a relatively pure environment by placing the cube right in the center of the two cameras, while the cube moved a large lateral displacement in the testing video. So, the images from the right camera in the testing video might become noises and lower the model performance of two-RGB-image-input, since the cross section of the cube caught by the right camera after large displacement could not align that from the left camera as well as in the training data set.

Secondly, the information offered by the image from the left camera may be enough to judge the object pose, so the disparity map cannot facilitate the prediction to perform better. And it even could become a distractor and offer noisy information during testing.
Besides, the patterns of the cube might be too complicated for the model. We may need to use an object with plainer colors, and so it can show the advantages of using the disparity map.

## Conclusion

In this project, we used two webcams to build the RGBD vision system to achieve a cheaper solution, and utilized the CNN structure to predict the object pose based on the images from the vision system. After setting up the two-webcam system, we built the training dataset by using a Rubik's Cube and rotating with different angles and took photos for it. Then, we constructed a CNN model and compare the performance by using three different image type of inputs. After training the model based on the dataset, results show that by inputting only one RGB image, it is good and stable enough to predict the object pose in 3D space.

However, this project has much more space to improve. Firstly, we can add the function of predicting object position in the structure, then this project can become more complete. Besides, we can use a robot arm to help us automate the procedure of building the training dataset in a more precise and efficient way. Lastly, since the model still cannot predict object pose precisely, it is possible to achieve more precise results that we can, at first, roughly predict the object pose based on CNN model, and then precisely predict the pose by using traditional methods, that is, a combinational structure.


## References
\[1\] Y. Xiang, T. Schmidt, V. Narayanan, and D. Fox, "PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes," arXiv:1711.00199v3, 2018.


{% include lib/mathjax.html %}
