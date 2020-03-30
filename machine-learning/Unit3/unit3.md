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

<img src="image1.png" width="589px"/>    

Neural signal transformation

<img src="image2.png" width="589px"/>  

#### Example Problem

Hidden layer representation: the next graph shows two linear combinations, after been evaluated by function $f$ of the circled point, which is positve for f2 and negative for f1

<img src="image3.png" width="589px"/>    

Now for a positive positive point, in the middle

<img src="image4.png" width="589px"/>  

Positive for f1 and negative for f2

<img src="image5.png" width="589px"/>  

Now evaluating all of the points,we get 

<img src="image6.png" width="589px"/>  

Notice that this is not linear separable. To make it separable we should apply some function in this case tanh

<img src="image7.png" width="589px"/>  

And with ReLu, which is not strictly separable

<img src="image8.png" width="589px"/>  

But if we flip the functions positive side, with the tanh function we get

<img src="image9.png" width="589px"/>  

And with ReLu, which makes it linear separable 

<img src="image10.png" width="589px"/>  

With random hidden units

<img src="image11.png" width="589px"/>  

With more randomly choosesn hidden units activations. Is it separable on the $R^{10}$ space? 

<img src="image12.png" width="589px"/>  

Actually it is

<img src="image13.png" width="589px"/>  

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

<img src="image14.png" width="589px"/>  

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

<img src="image15.png" width="589px"/>  

Looking as a simple example 

<img src="image16.png" width="589px"/>  

Now applying back propagation to our example

To change the variation of w1 that we have to apply, we need to find the derivative of Loss in function of w1. We can find it following the chain rule as the next image.

<img src="image17.png" width="589px"/>  

### Exercise

<img src="image18.png" width="589px"/>  

Let $n$ be the learning rate for the stochastic gradient descent algorithm.
Recall that our goal is to tune the parameters of the neural network so as to minimize the loss function. Which of the following is the appropriate update rule for the paramter 𝑤1 in the stochastic gradient descent algorithm? 

<img src="image19.png" width="589px"/>  

<img src="image22.png" width="589px"/>  

<img src="image20.png" width="589px"/>  

<img src="image21.png" width="589px"/>  

<img src="image23.png" width="589px"/>  

<img src="image24.png" width="589px"/>  

<img src="image25.png" width="589px"/>  

<img src="image26.png" width="589px"/>  

<img src="image27.png" width="589px"/>  

<img src="image28.png" width="589px"/>  

<img src="image29.png" width="589px"/>  

<img src="image30.png" width="589px"/>  

<img src="image31.png" width="589px"/>  

### Training Models with one hidden layer

With 2 hidden units

<img src="image32.png" width="589px"/>  

<img src="image33.png" width="589px"/>  

Wint 10 hidden units

<img src="image34.png" width="589px"/>  

Now other example of data

<img src="image35.png" width="589px"/>  

To 100 hidden units, the up left line from the next image should't be there because is not useful

<img src="image36.png" width="589px"/>  

Now when lines are initialized on the origin(left) and when they are initialized randomly

<img src="image37.png" width="589px"/>  

Size and optimization

<img src="image38.png" width="589px"/>  

<img src="image39.png" width="589px"/>  

#### Summary
* NN can be learned with SGD similarly to linear classifiers
* The derivatives nexessary for SGD can be evaluated effenctively via back-propagation
* Multi layer NN models are complicated ... we are no longer guaranteed to reach global (only local) optimun with SGD
* Larger model tend to be easier to learn ... units only need to be adjusted so that they are, collecively, sufficient to solve the task

## Lecture 10. Recurrent Neural Networks
Goals 
* Know the difference between feed-forward and recurrent neural networks(RNNs).
* Understand the role of gating and memory cells in long-short term memory (LSTM).
* Understand the process of encoding of RNNs in modeling sequences. 

### Introduction to RNN
#### Topics
* Modeling sequences 
* The problem of encoding sequences
* RNNs

#### Predict values based in historical (based on time) 

<img src="image40.png" width="589px"/>  

<img src="image41.png" width="589px"/>  

<img src="image42.png" width="589px"/>  

<img src="image43.png" width="589px"/>  

#### Learning to encode 

<img src="image44.png" width="589px"/>  

<img src="image45.png" width="589px"/>  

<img src="image46.png" width="589px"/>  

#### Encoding with RNN

<img src="image47.png" width="589px"/>  

<img src="image48.png" width="589px"/>  

<img src="image49.png" width="589px"/>  

#### Gating and LSTM (Long short term memory)

<img src="image50.png" width="589px"/>  

<img src="image51.png" width="589px"/>  

Key things
* NN for sequences: encoding
* RNNs, unfolded
    * state, evolution, gates
    * relation to feed-forward NN
    * back-propagation (conceptually)
* Issues: vanishing/exploding grandient
* LSTM (operationally)

## Recurrent Neural Networks
* Formulate, estimate and sample sequences from Markov models.
* Understand the relation between RNNs and Markov model for generating sequences.
* Understand the process of decoding of RNN in generating sequences. 
### Markov Models
Today we're going to be talking about how to genrate sequences using recurrent NN

* Modeling sequences: language models
    * Markov model
    * as NN
    * hidden state, RNN
* Example: decoding images into senteces

Description
* Next word in a sentence depends on previous symbols already written ( history =one, two, or more words)  
**The lecture leaves me bumfuzzled**  
* Similar, next character in a word depends on previous characters already written  
**bumfuzzled**
* We can model such kth order dependences between symbols with Markov Models

Markov Language Models  

<img src="image52.png" width="589px"/>  

<img src="image53.png" width="589px"/>  

Maximun likelihood estimation

<img src="image54.png" width="589px"/>  

<img src="image55.png" width="589px"/>  

### Markov Models to Feedforward Neural Nets
Feature baed Markov Model
* We can also represent the Markov model as a feedforward NN (very extendable)  

<img src="image56.png" width="589px"/>  

We can use two order Markov inserting past words (A trigram landuage model)

<img src="image57.png" width="589px"/>  

### RNN for sequences

<img src="image58.png" width="589px"/>  

<img src="image59.png" width="589px"/>  

Decoding, RNNs

<img src="image60.png" width="589px"/>  

<img src="image61.png" width="589px"/>  

### RNN Decoding
Decoding into a sentence

<img src="image62.png" width="589px"/>  

Mapping images to text

<img src="image63.png" width="589px"/>  

Examples

<img src="image64.png" width="589px"/>  

#### Key things
* Markov model for sequences
    * How to formulate, estimate, sample sequences from
* RNN for generation (decoding) sequencs
    * relation to Markov models
    * evolving hidden models
    * sampling from
* Decoding vectors into sequences
Help: [Activation Functions Graphs](https://adl1995.github.io/an-overview-of-activation-functions-used-in-neural-networks.html)
