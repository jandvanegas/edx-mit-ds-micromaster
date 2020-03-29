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

### Hidden layer models 
* $z_i=\sum_{j=1}^2 (x_j*w_{ji}+W_{0i})$
* $f_i=tanh(z_i)$

![](image1.png)

Neural signal transformation

![](image2.png)

#### Example Problem

Hidden layer representation: the next graph shows two linear combinations, after been evaluated by function $f$ of the circled point, which is positve for f2 and negative for f1

![](image3.png)

Now for a positive positive point, in the middle

![](image4.png)

Positive for f1 and negative for f2

![](image5.png)

Now evaluating all of the points,we get 

![](image6.png)

Notice that this is not linear separable. To make it separable we should apply some function in this case tanh

![](image7.png)

And with ReLu, which is not strictly separable

![](image8.png)

But if we flip the functions positive side, with the tanh function we get

![](image9.png)

And with ReLu, which makes it linear separable 

![](image10.png)

With random hidden units

![](image11.png)

With more randomly choosesn hidden units activations. Is it separable on the $R^{10}$ space? 

![](image12.png)

Actually it is

![](image13.png)

#### Summary
* Units in NN are linear classifiers, just with different output non-linearly
* The units in feed-forward neural networks are arranged in layers (input, hidden, .., output)
* By learnign the parameters associated with the hidden layer untis, we learn how to represent examples( as hiden layer activations)
* The representaion in NN are learned directly to facilitate the end-to-end task
* A simple classifier (output unit) suffices to solve comples classification tasks if it operates on the hidden layer representation

#### Exercise 
```
import numpy as np
import matplotlib.pyplot as plt
# Data
x = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
y = np.array([1, -1, -1, 1])
# Weights which change for each exercise
w_0 = np.array([[1, 1]])
w_0_g = np.tile(w_0.T, 4)
w = np.array([[1, -1], [-1, 1]])
# Outputs after the multiplication
z = w_0_g + np.matmul(w.T, x.T)

def relu(X):
    return np.maximum(0,X)
# Apply function to z. Try with tanh, relu and f(z)=z 
f_z = z

# Ploting
colors = ['red' if xi > 0 else 'green' for xi in y]
plt.scatter(f_z[0, :], f_z[1, :], c=colors)
plt.show()
```
Here is what we get for one of the weights

![](image14.png)

The goal is to test with which parameters the output of $f(z)$ becomes linear separable

## Feedforward Neural Networks, Back Propagation, and Stochastic Gradient Descent (SGD) 
### 1. Objectives
*  Write down recursive relations with back-propagation algorithm to compute the gradient of the loss function with respect to the weight parameters.
* Use the stochastic descent algorithm to train a feedforward neural network.
* Understand that it is not guaranteed to reach global (only local) optimum with SGD to minimize the training loss.
* Recognize when a network has overcapacity . 

### 2. Backpropagation 
* Learning feed-forward NN
* SGD and back-propagation

Learning NN

We want to update our weight values, base on the loss function gradient
![](image15.png)

Looking as a simple example 
![](image16.png)

Now applying back propagation to our example

To change the variation of w1 that we have to apply, we need to find the derivative of Loss in function of w1. We can find it following the chain rule as the next image.
![](image17.png)

### Exercise

![](image18.png)
Let 𝜂 be the learning rate for the stochastic gradient descent algorithm.
Recall that our goal is to tune the parameters of the neural network so as to minimize the loss function. Which of the following is the appropriate update rule for the paramter 𝑤1 in the stochastic gradient descent algorithm? 

![](image19.png)
![](image22.png)
![](image20.png)
![](image21.png)
![](image23.png)
![](image24.png)
![](image25.png)
![](image26.png)
![](image27.png)
![](image28.png)
![](image29.png)
![](image30.png)
![](image31.png)


