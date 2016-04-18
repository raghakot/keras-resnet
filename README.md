# keras-resnet
Residual networks implementation using Keras-1.0 functional API. 

### The original articles
 * [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385) (the 2015 ImageNet competition winner)
 * [Identity Mappings in Deep Residual Networks](http://arxiv.org/abs/1603.05027)

### Residual blocks
The residual blocks are based on the new improved scheme proposed in [Identity Mappings in Deep Residual Networks](http://arxiv.org/abs/1603.05027) as shown in figure (b)

![Residual Block Scheme](images/residual_block.png?raw=true "Residual Block Scheme")

Both bottleneck and basic residual blocks are supported. To switch them, simply provide the block function [here](https://github.com/raghakot/keras-resnet/blob/master/resnet.py#L109)

### Code walkthrough
The architecture is based on 50 layer sample (snippet from paper)

![Architecture Reference](images/architecture.png?raw=true "Architecture Reference")

There are two key aspects to note here

 1. conv2_1 has stride of (1, 1) while remaining conv layers has stride (2, 2) at the beginning of the block. This fact is expressed in the following [lines](https://github.com/raghakot/keras-resnet/blob/master/resnet.py#L91-L93).
 2. At the end of the first skip connection of a block, there is a disconnect in num_filters, width and height at the merge layer. This is addressed in [`_shortcut`](https://github.com/raghakot/keras-resnet/blob/master/resnet.py#L69) by using `conv 1X1` with an appropriate stride. For remaining cases, input is directly merged with residual block as identity.

### 50 Layer resnet model sample
Generated 50 Layer resnet model visualization [here](https://github.com/raghakot/keras-resnet/blob/master/images/resnet_50.png)

