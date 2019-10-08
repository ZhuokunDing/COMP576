# Introduction to Deep Machine Learning

## Why do we need Deep Learning

How to deal with nuisance variations: build a **selective**, **invariant** and **multi-task** representation to achieve disentanglement.

## Definition of Neural Networks

- takes in inputs and returns outputs
- layers of processing: alternates between linear and nonlinear transformations typically
- Can be trained to learn complex functions
- Inspired by the brain

## Object recognition

* transfer learning

## Facial recognition

## Deep Art

Combining content and style from different images.  

Different layers of the deep convnet extract different levels of correlations in images:

- earlier layer: local representations
- later layer: global representations

## Self-Driving Cars

Manually decompose the auto-driving problems and implement individual neural networks for decomposed aspects of the problem.

## Generative models for Natural Images

Why GANs generate horrible dogs: networks do not require large global scale correlations to differentiate dogs.

## Generative Adversarial Nets (GANs) for Natural Image Translation

## Formal Language Representation in trained NNs

Networks(BERT) trained to perform the fill-in-the-blank task learn the parsing of natural language.

Networks trained to predict the next character of a string could learn the full grammar and syntax of the language.

Why:
$$
\begin{align}
P(x_1, \dots ,x_T)&=p(x_1)p(x_2|x_1)\dots p(x_T|x_1,\dots,x_{T-1})\\
&\approx \Pi^{T}_{t=1}p(x_t|h(x_t))

\\
Max_{\theta}(ln\ P(x_1,\dots,x_T|\theta)) &= \frac{1}{\tau}\Sigma^T_{t=1}ln\ p(x_t|RNN(x_{t\leq T});\theta)
\end{align}
$$
where $h(x_t)$ is a low dimensional summary of the language grammar.