# Device Placement Optimization with Reinforcement Learning

This project is part of CSE 5429 (Hardware Accelerators for Machine Learning) and aims to develop a Neural Machine Transalation model to map the nodes of operations in graph models to various devices such as GPUs and CPUs.

We have tried to implement the latest technique proposed by Google. The following link leads to their research paper (https://arxiv.org/abs/1706.04972)

mnist_placement.py : This code is based on the basic MNIST model provided in TensorFlow tutorials. We have modified the code to create seperate functions for our convenience such as make_seesion, apply_placement and train.

