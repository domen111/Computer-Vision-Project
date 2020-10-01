Object State Tracking in 3D Space by Using Two Webcams
===

## Introduction
6D object pose estimation in real time, including location and orientation, is an important technique in many fields. Such helpful information can facilitate many attractive ongoing researches, such as robotic manipulation tasks and autonomous self-driving cars. However, most of the existing state-of-the-art papers for object pose estimation depend on a lot of labeled training data to conduct supervised learning. Besides, most of the current research utilized RGB-D or point-cloud cameras, which are expensive and hard to access for ordinary people. So in this project, we try to use a 3D deep auto-encoder (DAE) with spatial softmax [1] to get the 6D pose of the target object, which only requires few vision data without annotation in unsupervised learning. Furthermore, the algorithm will depend on the RGB-D images, which will be generated from a custom-made vision system composed of two inexpensive commercial 2D webcams. 

## Proposal
[Project Proposal](./proposal.pdf)