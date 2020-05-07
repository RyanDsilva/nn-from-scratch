# Neural Networks From Scratch

🌟 Implementation of Neural Networks from Scratch Using Python &amp; Numpy 🌟

<p align="center">
  <img src="images/nn.webp" width="550px">
</p>

> Uses Python 3.7.4

#### This repository has detailed math equations and graphs for every feature implemented that can be used to serve as basis for greater, in-depth understanding of Neural Networks

## Contents 📑

- [Core Concepts](./core)
- [Activation Functions](./activations)
- [Loss Functions](./loss)
- [Optimizers](./optimizers)

## Setup 💻

```bash
git clone <url>
pip install -r requirements.txt
```

Here, Keras is used just to load the MNIST dataset

## Usage 📔

- Tune hyperparameters in `config.py`
- Run the following command

```bash
python main.py
```

#### Output:

<pre>
$ python main.py
epoch 1/30      error=0.230924
epoch 2/30      error=0.099688
epoch 3/30      error=0.082054
epoch 4/30      error=0.070288
epoch 5/30      error=0.061518
.....
.....
.....
epoch 25/30     error=0.014255
epoch 26/30     error=0.013660
epoch 27/30     error=0.013108
epoch 28/30     error=0.012597
epoch 29/30     error=0.012061
epoch 30/30     error=0.011587

Predicted Values:
[array([[ 0.02636159, -0.00880932, -0.06221383,  0.0892342 , -0.030511  ,
        -0.10164595,  0.01672581,  <b>0.96746597</b>, -0.01456723,  0.15428294]]), 
array([[ 0.35812139, -0.00917876,  <b>0.42699824</b>,  0.06618954,  0.02116222,
        -0.14526985,  0.05195041,  0.14218301,  0.33999944, -0.18716734]])]
True Values:
[[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]]
</pre>

## Roadmap 📑

- [x] Basic Activation Functions
- [x] Basic Loss Functions
- [x] Gradient Descent
- [x] Layer Architecture
- [x] Wrapper Classes
- [x] Hyperparameters Configuration
- [ ] Exotic Functions
  - [ ] SoftMax Activation
  - [ ] Gradient Descent w/ Momentum
  - [ ] RMSProp Optimizer
  - [ ] Adam Optimizer
  - [ ] CrossEntropy Loss Function
  - [ ] GeLu Activation
- [ ] Regularization
- [ ] Clean Architecture
- [ ] UI (Similar to Tensorflow Playground)

##### This project is not meant to be production ready but instead serve as the foundation repository to understand the in-depth working of Neural Networks down to the mathematics of the task.

###### Collaborations in implementing and maintaining this project are welcome. Kindly reach out to me if interested.

## References 📚

- Deep Learning Specialization, Andrew NG - Coursera
- [Machine Learning Cheatsheet](https://ml-cheatsheet.readthedocs.io/en/latest/index.html)

> &copy; 2020 Ryan Dsilva
