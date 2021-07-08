# Implementation of a leaky cell for PyTorch

![equation](http://latex.codecogs.com/gif.image?\dpi{110}&space;h_{t}&space;=&space;\sigma&space;((1-\lambda)h_{t-1}&space;&plus;&space;\lambda&space;(W_{in}x_{t}&space;&plus;&space;W_h&space;h_{t-1})&space;))

where:
- h is the state of the leaky cell
- x is the input
- lambda is the leak rate
- Win is the input weight matrix
- Wh is the hidden weight matrix
- sigma is the nonlinearity or transfer function, typically, hyperbolic tangent

Includes:
----------
- a leaky cell module that can be used as a recurrent layer in any architecture
- a function to generate initial weights for the leaky cell
- a simple one layer leaky recurrent network trainable with backprop
- a reservoir implementation with a specific 'train' method to train the readout weights only
