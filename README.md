## Build Your deep learning framework
Construct your own deep learning framework,which help you to be familiar with **forward propagation, backward propagation, and update model parameters by gradient descent method**. This is a very simple deep learning framework, which is conducive to understand model and optimization.
### Function
* SGD_Linear_regression: We use Least-Squares method and the SGD optimization model to perform linear regression,and compare the difference between them.
* LearnConv_TraditionCV: We use the traditional method to transform the color of the image. At the same time, we use the network to learn the color transformation, and compare the difference between the two.

### Code Structure
#### Conv and Fc
 For every layer, we need to code the forward function and the backward function. 
 * the formula doesn't errors
 * the relationship between dimensions is not wrong(Do not use reshape at will).

#### optim
Method of updating parameters based on weight and gradient.

#### model
This class has the following main effects:
* Contact forward and backward(cache)
* store the parameters of model


#### solver
This class has the following main effects:
* Save the parameters needed for optimization,  such as the momentum of each variable, the order of magnitude of each variable.
