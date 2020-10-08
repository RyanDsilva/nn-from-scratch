# Neural Networks From Scratch

🌟 Implementation of Neural Networks from Scratch Using Python &amp; Numpy 🌟

> Uses Python 3.7.4

## Activation Functions

Activation functions live inside each neuron in the neural network layers and modify the data they receive before passing it to the next layer. Activation functions give neural networks their power  by  introducing non-linearity, allowing them to model complex relationships.


`x: input`

- Identity

    <img src="images/identity_eq.png" title="f(x) = x" />
    <br>
    <img src="images/identity.png" height="200px">


- BinaryStep

    <img src="images/binarystep_eq.png" title="f(x) = \begin{cases} 0 & \text{ if } x<= 0 \\ 1 & \text{ if } x>0 \end{cases}" />
    <br>
    <img src="images/binarystep.png" height="200px">

- Linear

  <img src="images/linear_eq.png" title="f(x) = cx" />
  <br>
  <img src="images/linear.png" height="200px">

- Sigmoid

  <img src="images/sigmoid-eq.png" title="f(x) = \frac{1}{1+e^-^x}" />
  <br>
  <img src="images/sigmoid.png" height="200px">

- Hyperbolic Tangent (tanh)

  <img src="images/tanh-eq.png" title="f(x) = \frac{e^x-e^-^x}{e^x+e^-^x}" />
  <br>
  <img src="images/tanh.png" height="200px">

- Rectified Linear Units (ReLU)

  <img src="images/relu-eq.png" title="f(x) = \begin{cases} 0 & \text{ if } x<= 0 \\ x & \text{ if } x>0 \end{cases}" />
  <br>
  <img src="images/relu.png" height="200px">

- Leaky Rectified Linear Units (LeakyReLU)

  <img src="images/leaky_relu-eq.png" title="f(x) = \begin{cases} bx & \text{ if } x<= 0 \\ x & \text{ if } x>0 \end{cases}" />

  `where b is a small constant`

  <br>
  <img src="images/leaky_relu.png" height="200px">

- Softmax

  <img src="images/softmax-eq.png" title="softmax(x)_i = \frac{exp(x_i)}{\sum_{j}^{ }exp(x_j))}" />
  <br>

- Gaussian Error Linear Units (GeLU)

  `err: error function`

  <img src="images/gelu-eq.png" title="f(x) = \frac{x(1+err(\frac{x}{\sqrt{2}}))}{2}" />
  <br>
  <img src="images/gelu.png" height="200px">
