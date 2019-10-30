# CNN

## Activation Function

### ReLU:

​	Some neurons might not have any gradient in backprop. Leaky ReLU is designed to avoid those situations. ELU is a differentiable approximation of ReLU. 

1. does not saturate
2. very computationally efficient
3. converge much faster
4. not zero-centered
5. dead in-regions, since relu is zero at negative values

### Sigmoid:

	1. saturated neurons kill the gradient
 	2. sigmoid outputs are not zero-centered, might lead to gradient explosion
 	3. exp() is a bit expensive

### Tanh:

1. saturating

## Preprocessing

approximately whitening the data by zero-center and normalize, but not rotating.

Why not decorrelate features in data like PCA: too costly

## Data Augmentation

reduces overfitting

data augmentation would violate independence between samples 

## Weight Initialization

Do not initialize all weight to zero, or all gradients would be zero.

### Xavier initialization:

​	fix the vanishing activation problem,  but it breaks when using ReLU nonlinearity

Data dependent initializations could help a lot for some non-convex machine learning tasks, like clustering (k-means initialization)

## Training Algorithms

### SGD

## Annealing Learning Rate

http://cs231n.github.io/neural-networks-3/#sgd

## Regularization

### Drop out

set a subset of neurons to zero in the forward pass. Avoiding overfit by reducing numbers of parameters. 

Another interpretation is that by drop out, we are actually training a large ensemble of models. Each binary mask is one model and trained on only ~one datapoint.

Drop out should behavior differently during training and test.

### Batch Norm

## Hyperparameter Optimization

* network architecture
* learning rate, decay schedule, update type
* regularization



