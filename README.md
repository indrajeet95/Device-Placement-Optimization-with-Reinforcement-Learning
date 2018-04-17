# Device Placement Optimization with Reinforcement Learning

This project is part of CSE 5429 (Hardware Accelerators for Machine Learning) and aims to develop a Neural Machine Transalation model to map the nodes of operations in graph models to various devices such as GPUs and CPUs.

We have tried to implement the latest technique proposed by Google. The following link leads to their research paper (https://arxiv.org/abs/1706.04972). The problem can be pictured using the tutorial provided by Google (https://www.tensorflow.org/tutorials/seq2seq). By reading the previous link, we can understand that the source words are English and target words are Vietnamese. For our problem, the english words will be set of operations defined in the Tensorflow graph and the target words will be set of devices, for example, CPU0, CPU1, GPU0, GPU1, etc.

mnist_placement.py : This code is based on the basic MNIST model provided in TensorFlow tutorials. We have modified the code to create seperate functions for our convenience such as make_seesion, apply_placement and train.

seq2seq_unroll.py: This part of the code defines the Neural Machine Transalation Model and also takes in a vocab file as input which contains the set of devices on which the benchmark will be running. The benchmark in our case is the mnist model. The code trains NMT model with runtime of MNIST model serving as reward.
