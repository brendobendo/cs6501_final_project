# Estimating Meal Calorie Counts Using Deep Learning and Regression

## Summary
 Manually logging food and calorie intake can be significantly time-consuming. In this paper, we present a far simpler approach to calorie tracking. We propose that calorie counts can be estimated from an image alone. That is, given a picture of a meal, our model can detect which foods are in the image and estimate the meal's aggregate calorie count. 
 
 ![The idea in a nutshell](https://github.com/brendobendo/cs6501_final_project/blob/master/intro_pic.jpg)
 
 To test this hypothesis, we first trained a deep convolutional network for large-scale image recognition (VGG) to classify over 221 different types of foods. We achieved over 75\% MAP, demonstrating that food classification is highly effective with state of the art CNN architectures. Then, as a proof of concept, we trained a faster-rcnn type neural net on a 10-class food data-set. The model then performs a calorie lookup for its top class predictions and sums these  counts to give a final calorie estimation. Our model achieved over 72\%  MAP for object detection and the calorie predictions were reasonably close to our test cases where total calorie counts were known. Our contributions to the field are twofold. First, we have created the first publicly available food-detection dataset that is adequately sized for deep learning. Second, we will show in future work that is is possible to predict calories by regressing on image features. For a more information about our results and methodology, please see the [project report](https://github.com/brendobendo/cs6501_final_project/project_report.pdf).
 

## Software Requirement:
We wrote all our code in Python 2.7. The code depends on the following libraries: 
1. Pytorch
2. Python 2.7
3. PIL
4. matplotlib
5. [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch)
6. tensorflow

## Useful Materials
This project aims to promote understanding of deep learning for computer vision and object recognition. For those interested in learning about the theory behind these networks, here are some great papers and resources to get started. 

1. [**r-cnn**](https://arxiv.org/pdf/1504.08083.pdf)
2. [**fast-rcnn**](https://arxiv.org/pdf/1504.08083.pdf)
3. [**faster-rcnn**](https://arxiv.org/pdf/1506.01497.pdf)
4. [**YOLO**](https://arxiv.org/pdf/1506.02640.pdf)
5. [**Visualizing and Understanding Convolutional Networks**](https://arxiv.org/pdf/1311.2901.pdf)
6. [**Mask-rcnn**](https://arxiv.org/pdf/1703.06870.pdf)

For the sake of computational efficiency and time limitations, we used Faster-rcnn to perform object detection for this project. And supplement materials helping understand region-cnn is available [here](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/object_localization_and_detection.html).


## Impletation of Faster-rcnn and other methods
A number of faster-rcnn impletation is available online, including:
- [The matlab code of original paper](https://github.com/ShaoqingRen/faster_rcnn).
- [Pycaffee version faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)
- [Pytorch version faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn)
- [Python chainer version faster-rcnn](https://github.com/chainer/chainercv)
- [Pytorch version yolo](https://github.com/marvis/pytorch-yolo2)
- [Pytorch version Mask R-CNN](https://github.com/felixgwu/mask_rcnn_pytorch)
- [Pytorch tutorial](https://github.com/ritchieng/the-incredible-pytorch)

## Pre-train vgg16 before object dection
```
cd ./tools
python trainvgg16.py --pretrained --gpu --checkPoint <file name>
```
And model parameters will be saved in the same folder


## Visualize net
We use tensorboardX to visualize our model.
```
cd ./tools
python visualize vgg16.py --model_dir <folder you saved your model>
tensorboard --logdir runs
```
or you can do:
```
cd ./data/saved_model
tensorboard --logdir runs
```
## Snapshot of vgg16 on validation data
```
cd ./tools
python vggSnippet.py --model_fir <folder you saved your model> --valData <root of your validation data>
```



