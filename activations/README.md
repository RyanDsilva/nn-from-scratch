# Neural Networks From Scratch

🌟 Implementation of Neural Networks from Scratch Using Python &amp; Numpy 🌟

> Uses Python 3.7.4

## Activation Functions

- Linear

  <img src="https://latex.codecogs.com/png.latex?f(x)&space;=&space;x" title="f(x) = x" />
  <br>
  <img src="images/linear.png" height="200px">

- Sigmoid

  <img src="https://latex.codecogs.com/png.latex?f(x)&space;=&space;\frac{1}{1&plus;e^-^x}" title="f(x) = \frac{1}{1+e^-^x}" />
  <br>
  <img src="images/sigmoid.png" height="200px">

- Hyperbolic Tangent (tanh)

  <img src="https://latex.codecogs.com/png.latex?f(x)&space;=&space;\frac{e^x-e^-^x}{e^x&plus;e^-^x}" title="f(x) = \frac{e^x-e^-^x}{e^x+e^-^x}" />
  <br>
  <img src="images/tanh.png" height="200px">

- Rectified Linear Units (ReLU)

  <img src="https://latex.codecogs.com/png.latex?f(x)&space;=&space;\begin{cases}&space;0&space;&&space;\text{&space;if&space;}&space;x<=&space;0&space;\\&space;x&space;&&space;\text{&space;if&space;}&space;x>0&space;\end{cases}" title="f(x) = \begin{cases} 0 & \text{ if } x<= 0 \\ x & \text{ if } x>0 \end{cases}" />
  <br>
  <img src="images/relu.png" height="200px">

- Leaky Rectified Linear Units (LeakyReLU)

  <img src="https://latex.codecogs.com/png.latex?f(x)&space;=&space;\begin{cases}&space;bx&space;&&space;\text{&space;if&space;}&space;x<=&space;0&space;\\&space;x&space;&&space;\text{&space;if&space;}&space;x>0&space;\end{cases}" title="f(x) = \begin{cases} bx & \text{ if } x<= 0 \\ x & \text{ if } x>0 \end{cases}" />

  `where b is a small constant`

  <br>
  <img src="images/leaky_relu.png" height="200px">

- Softmax

  <img src="https://latex.codecogs.com/png.latex?f(x)_i&space;=&space;\frac{exp(x_i)}{\sum_{j}^{&space;}exp(x_j))}" title="softmax(x)_i = \frac{exp(x_i)}{\sum_{j}^{ }exp(x_j))}" />
  <br>

- Gaussian Error Linear Units (GeLU)

  <img src="https://latex.codecogs.com/png.latex?f(x)&space;=&space;\frac{x(1&plus;err(\frac{x}{\sqrt{2}}))}{2}" title="f(x) = \frac{x(1+err(\frac{x}{\sqrt{2}}))}{2}" />
  <br>
  <img src="images/gelu.png" height="200px">
