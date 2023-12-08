# Implementing Deep Learning Architectures for Advanced Machine Learning using PyTorch

This repository contains my implementations of Deep Neural Network based architectures for specific usecases using PyTorch. Each implementation comes with a detailed problem specification and a link to the solution.

## EE 782 : Advanced Topics in Machine Learning

### Course Instructor : Prof. Amit Sethi

## Long-Short-Term-Memory (LSTM) based Algorithmic Stock Trader
- Given the [sp500 stock market tickers](https://www.kaggle.com/datasets/rohanrkalbag/ee782) dataset
- Modelling the time series using LSTM, and perform various experiments involving techniques like Normalization, Feature Engineering
- Assessing the profitability of the algorithmic trading module under various conditions like buy-ask spread, commissions .etc
- Detailed problem specification can be found [here](https://github.com/rohankalbag/advanced-machine-learning/blob/main/assignment-1/EE782%202023%20A1.pdf) 
- My solution for the same can be found [here](https://github.com/rohankalbag/advanced-machine-learning/blob/main/assignment-1/EE782_A1_20d170033.ipynb)

<img width="602" alt="" src="https://github.com/rohankalbag/advanced-machine-learning/assets/46604893/9ea74402-634e-45bc-a592-e15580d0f067">

![output](https://github.com/rohankalbag/advanced-machine-learning/assets/46604893/7211cb75-de01-4d29-93d5-8951df6befaa)

## Facial Similarity Metric Learning and Face Generation using Deep Convolutional Generative Adversarial Networks (DCGAN)
- Given the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) dataset
- Using a Transfer Learned ResNet based Siamese Network for Similarity Metric Learning for Identification
- Performing experiments using techniques like Regularization, Learning Rate Scheduling, Dropout, Variation of Optimizers to identify best performing model
- Training a DCGAN to generate new faces from input gaussian noise
-  Modifying the DCGAN to become a Conditional GAN, by using the Siamese Network trained earlier, i.e, given an input image generate an unseen image of the same person
- Detailed problem specification can be found [here](https://github.com/rohankalbag/advanced-machine-learning/blob/main/assignment-2/EE782%202023%20A2.pdf) 
- My solution for the same can be found [here](https://github.com/rohankalbag/advanced-machine-learning/blob/main/assignment-2/EE782_A2_20d170033.py)

![faces](https://github.com/rohankalbag/advanced-machine-learning/assets/46604893/62255958-3141-48d4-a098-9b648590921c)

Input Image    |  Conditionally Generated Image 
:-------------------------:|:-------------------------:
<img src="https://github.com/rohankalbag/advanced-machine-learning/assets/46604893/720f9817-a669-437a-87de-bba59956b860" width=455></img> | <img src="https://github.com/rohankalbag/advanced-machine-learning/assets/46604893/2205ea0f-0a42-4702-8e12-86e003e70186" width=425></img> 

