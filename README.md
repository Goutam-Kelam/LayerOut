## Official repository of [LayerOut: Freezing Layers in Deep Neural Networks](https://link.springer.com/article/10.1007/s42979-020-00312-x)
Deep networks involve a huge amount of computation during the training phase and are more prone to overfitting. To ameliorate these, several techniques such as DropOut, DropConnect, Stochastic Depth, and BlockDrop have been proposed. These techniques regularize a neural network by dropping nodes, connections, layers, or blocks within the network. However, their functionality is limited in that they are suited for only fully connected networks or ResNet-based architectures. In this paper, we propose LayerOut - a regularization technique. This technique can be applied to both fully connected networks and all types of convolutional networks such as, VGG-16, ResNet etc. In LayerOut, we stochastically freeze the trainable parameters of a layer during an epoch of
training. Our experiments on MNIST, CIFAR-10, and CIFAR-100 show that LayerOut generalizes well and reduces the computational burden significantly. In particular, we have observed up to 70% reduction, on an average, and in an epoch, in computation and up to 2% enhancement in accuracy as compared to baseline networks.



## Cite

    @article{goutam2020layerout,
      title={LayerOut: Freezing Layers in Deep Neural Networks},
      author={Goutam, Kelam and Balasubramanian, S and Gera, Darshan and Sarma, R Raghunatha},
      journal={SN Computer Science},
      volume={1},
      number={5},
      pages={1--9},
      year={2020},
      publisher={Springer}
    }
