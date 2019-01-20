# LayerOut
Deep networks involve a huge amount of computation during
the training phase and are more prone to overfitting. To ameliorate these,
several techniques such as DropOut, DropConnect, Stochastic Depth,
and BlockDrop have been proposed. These techniques regularize a neu-
ral network by dropping nodes, connections, layers, or blocks within the
network. However, their functionality is limited in that they are suited
for only fully connected networks or ResNet-based architectures. In this
paper, we propose LayerOut - a regularization technique. This technique
can be applied to both fully connected networks and all types of convolu-
tional networks such as, VGG-16, ResNet etc. In LayerOut, we stochas-
tically freeze the trainable parameters of a layer during an epoch of
training. Our experiments on MNIST, CIFAR-10, and CIFAR-100 show
that LayerOut generalizes well and reduces the computational burden
significantly. In particular, we have observed up to 70% reduction, on an
average, and in an epoch, in computation and up to 2% enhancement in
accuracy as compared to baseline networks.
