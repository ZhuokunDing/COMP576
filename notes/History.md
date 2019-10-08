# History of Deeplearning

express power VS learning ability

## McCulloch-Pitts Neurons (1943)

### Assumptions:

1. All or none
2. A fixed threshold
3. The only significant delay is the synaptic delay
4. Inhibition absolutely kills excitation
5. The structure of the net does not change with time

<img src="C:\Users\Zhuokun\AppData\Roaming\Typora\typora-user-images\1567625197896.png" alt="1567625197896" style="zoom: 33%;" />

### McCulloch-Pitts Nets: 

A McCulloch-Pitts Net is a directed graph G with McCulloch-Pitts Neurons as nodes and edges marked as either excitatory or inhibitory.

If G is acyclic, it is a **feed-forward** net. Otherwise, it is a **recurrent** net.

### Expressive power

Feed-forward McCulloch-Pitts nets can compute any Boolean function

Recursive McCulloch-Pitts net can simulate any deterministic finite automation.

## Problems/limitations of expressive power:

Mathematical definition of expressive power:
$$
\forall \epsilon,\ \forall f \in F_{nice} ,\ \exist \  \theta,\ such\ that\ ||f-\hat f_\theta||_{L^p}\leq\epsilon
$$
Learnability is not implied.

## Hebb's Postulate

<img src="C:\Users\Zhuokun\AppData\Roaming\Typora\typora-user-images\1567629029567.png" alt="1567629029567" style="zoom:33%;" />

## The Perceptron

