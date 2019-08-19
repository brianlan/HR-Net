# HR-Net

A Simple PyTorch implementation based on paper [High-Resolution Representations for Labeling Pixels and Regions](https://arxiv.org/abs/1904.04514)
Did some modifications on the ResBlock to save computation. 

This repo is only a demo of showing the implementation is basically working...

### Input Image
I only labeled two classes: car and lane line.
![Input](https://github.com/brianlan/HR-Net/blob/master/data/night.jpg?raw=true)


### Evaluation Results
The model is trained on a single image (i.e. the image above), and also predict the result of it.
This is so called over-fitting.

![car](https://github.com/brianlan/HR-Net/blob/master/results/car.png?raw=true)

![lane](https://github.com/brianlan/HR-Net/blob/master/results/lane.png?raw=true)
