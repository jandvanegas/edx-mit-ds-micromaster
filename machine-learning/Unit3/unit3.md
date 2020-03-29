# Unit 3
Goals 
* Feedforward neural networks
* Gradient of the loss function using back propagation
* Recurrent Neural Networkd (RNNs) and long-short-term memory (LSTM)
* Convolutional neural networks

## Introduction to Feedforward Neural Networks

Goals:
* Layers in a feedforward neural network
* Activation Functions (eg. tang, ReLu)
* Hidden Layers
* Linear separation

Quiz:  
If $\hat{y}=sign(\theta \phi(x))$ 
Then:
* The feature map $\phi$ is not always function from $R^d$ to $R^d$
* But if $\phi(x) \in R^d$ the the classification $\theta$ is also a vector $R^d$ 

The difference from NN and other non linear methods is:
* In NN, the parameters and the feature representation ($\phi$), both has to be tunned 

### Neural Networks Units
Looking real Neurons, they have 
* Dendrites
* Axon

And the abstraction of it is a linear classifier. It has inputs (Dendrites) $X$, weights $\theta$ and output function $f$ (Axon) 

We have some types of NN
* Linear
* ReLu $f(x) = max(0,x)$
* $tanh(z)=(e^z-e^{-z})/(e^z+e^{-z})$ which is an odd function, with range from -1 to 1

Features NN
* Loosely motivated by biological neurons, network
* adjustable processing units
* highly parallel
* deep = many transformations (layers) before output

Deep Learning
* It has overtaken a number of academid disciplices in just a few years
* It has applications on
	* CV
	* NLP
	* Speech recognition
	* Computational Biology, etc
* Key role in recent successes
	* self driving vehicles
	* conversational agents
	* super human game playing

Why so popular
1. Lots of data
1. Computational resources
1. Large models are easier to train
	1. Flexible neural "lego pieces"

One hidden layer model
With two layers and two nodes each one
* $z_i=\sum_{j=1}^2 (x_j*w_{ji}+W_{0i})$
* $f_i=tanh(z_i)$
