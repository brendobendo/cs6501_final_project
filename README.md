# cs6501_final_project--Counting Calories of a Meal Using Deep Learning and Regression

## Problem   Description/Motivation

Logging food and calorie intake has been shown to facilitate weight management. Many smartphone apps allow the user to track calories by manually specifying the food types and portion sizes eaten at each meal, which is significantly time-consuming and cumbersome. We propose a system that is able to estimate calorie counts from an image alone. This system possesses numerous advantages to manual calorie tracking. First, it is far more natural; between writing yelp reviews and snapchat, photograhing meals has become all too common for millennials and young adults. Second, it will drastically reduce the required time and burden of calore-tracking. It is our hoep that our system will promote users to be more conscious of their diets and ultimately become healthier eater.

## Software Requirement:
1. Pytorch
2. Python 2.7
3. PIL
4. matplotlib

## Useful Materials
This project aims to promote my understanding of vision computation and object recognition. There are extensive research focused on object detection:
1. [**r-cnn**](https://people.eecs.berkeley.edu/~rbg/papers/r-cnn-cvpr.pdf)
2. [**fast-rcnn**](https://arxiv.org/pdf/1504.08083.pdf)
3. [**faster-rcnn**](https://arxiv.org/pdf/1506.01497.pdf)
4. [**YOLO**](https://arxiv.org/pdf/1506.02640.pdf)

For the sake of computational efficiency and time limitation, we concentrate on using Faster-rcnn to perform object detection. And supplement materials helping understand region-cnn is available [here](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/object_localization_and_detection.html).
