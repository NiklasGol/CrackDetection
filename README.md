# Crack_detect

## Description

This code can be used in order to train and evaluate a neural network
(e.g. ResNet18, ResNet50) on images with binary classification. It was used on
an image data set [[1]](#1) with 40000 labeled concrete images.
In order to enhance the performance the training pipeline features data
augmentation to further increase amount of training data and get bigger variety.
Lastly vanilla gradient and guided backpropagation are used to get insights on
the decision process of the network.

## How to use

1. Use train.py to train the network.
2. To visualize the training process visualize_training.py can be used.
3. Bigger images can be processed in a sliding window approach via the
sliding_window.py script.
4. To obtain classification and saliency maps use use_model.py.

## Examples
![alt text](./examples/sw.gif)

## References
<a id="1">[1]</a>
Caglar Özgenel and Arzu Sorguc (2018).
Performance Comparison of Pretrained Convolutional Neural Networks on Crack Detection in Buildings.
DOI:10.22260/ISARC2018/0094.
