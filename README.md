# cs6501_final_project--Counting Calories of a Meal Using Deep Learning and Regression

## Problem   Description/Motivation

Logging food and calorie intake has been shown to facilitate weight management. Many smartphone apps allow the user to track calories by manually specifying the food types and portion sizes eaten at each meal, which is significantly time-consuming and cumbersome. We propose a system that is able to estimate calorie counts from an image alone. This system possesses numerous advantages to manual calorie tracking. First, it is far more natural; between writing yelp reviews and snapchat, photograhing meals has become all too common for millennials and young adults. Second, it will drastically reduce the required time and burden of calore-tracking. It is our hoep that our system will promote users to be more conscious of their diets and ultimately become healthier eater.

## Software Requirement:
1. Pytorch
2. Python 2.7
3. PIL
4. matplotlib
5. [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch)

## hardware requirement
- The usage of GPU is highly recommended.
- Cloud computing services are available. For instance, [AWS](https://aws.amazon.com/free/?sc_channel=PS&sc_campaign=acquisition_US&sc_publisher=google&sc_medium=cloud_computing_hv_b&sc_content=aws_core_e_control_q32016&sc_detail=aws&sc_category=cloud_computing&sc_segment=188908133959&sc_matchtype=e&sc_country=US&s_kwcid=AL!4422!3!188908133959!e!!g!!aws&ef_id=V4HLzAAAAcKEocf1:20171017234041:s).

## Useful Materials
This project aims to promote my understanding of vision computation and object recognition. There are extensive research focused on object detection, here is a snapshot:
1. [**r-cnn**](https://arxiv.org/pdf/1504.08083.pdf)
2. [**fast-rcnn**](https://arxiv.org/pdf/1504.08083.pdf)
3. [**faster-rcnn**](https://arxiv.org/pdf/1506.01497.pdf)
4. [**YOLO**](https://arxiv.org/pdf/1506.02640.pdf)
5. [**Visualizing and Understanding Convolutional Networks**](https://arxiv.org/pdf/1311.2901.pdf)
6. [**Mask-rcnn**](https://arxiv.org/pdf/1703.06870.pdf)

For the sake of computational efficiency and time limitation, we concentrate on using Faster-rcnn to perform object detection. And supplement materials helping understand region-cnn is available [here](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/object_localization_and_detection.html).

## Impletation of Faster-rcnn and other methods
A number of faster-rcnn impletation is available online, including:
- [The matlab code of original paper](https://github.com/ShaoqingRen/faster_rcnn).
- [Pycaffee version faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)
- [Pytorch version faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn)
- [Python chainer version faster-rcnn](https://github.com/chainer/chainercv)
- [Pytorch version Mask R-CNN](https://github.com/felixgwu/mask_rcnn_pytorch)
- [Pytorch tutorial](https://github.com/ritchieng/the-incredible-pytorch)

## Plan
- [x] **Data Preprocessing**:
Merge [food 101 dataset](https://www.kaggle.com/kmader/food41/data) with [Vireo-Food_172 dataset](http://vireo.cs.cityu.edu.hk/VireoFood172/).
- [ ] **Training pre-trained VGG**:
more details needed



