# Implementation of a leaky cell for PyTorch

Includes:
----------
- a leaky cell module that can be used as a recurrent layer in any architecture
- a function to generate initial weights for the leaky cell
- a simple one layer leaky recurrent network trainable with backprop
- a reservoir implementation with a specific 'train' method to train the readout weights only
