# Week 2 Notes

## Case studies

* LeNet-5
* AlexNet
* VGG

Laid foundations for modern CV.

* ResNet (152)
* Inception

## Classic Networks

You shuold read them in the order: AlexNet (easy), VGG, LeNet(hard)

### LeNet-5

* digit recognition
* in that age everyone used "valid" convolutions
* Essentially 4 conv -> flatten -> fc -> fc -> logits / softmax $\hat{y}$

Architecture:

![lenet5_arch](lenet5_arch.png)

* Not unusual to see networks that are 1000 times bigger
* width / height goes down, number of channels goes up

`conv pool conv pool .... fc fc fc .. output`

#### Extra comments if you want to read LeNet-5

* use sigmoid/tanh instead of ReLU
* typically every filter looks at every channel, but original LeNet-5 will not do this, it will look at single channels and complex stuff because the network was already "big" back then
* original LeNet-5 uses sigmoid nonlinearity after pooling.

![lenet5_arch_extended](lenet5_arch_extended.png)

### AlexNet

* Original paper uses 224x224x3 but its actually 227x227
* Uses large stride, so dim shrinks quickly
* alot bigger than LeNet-5
* 8 convs, flatten, 3 fc, softmax
* very similar to LeNet-5
    * 60k params (LeNet) vs 60mil params (AlexNet)
* uses ReLU activation function

Architecture:

![alexnet_arch](alexnet_arch.png)

#### Extra comments if you want to read AlexNet

* uses complex training schedule on two GPU's
* uses Local Response Normalization
    * not used much
    * basic idea: looks at 1 position in H,W and look through all channels and normalize them. Inituition is that for each position in this 13x13 image you dont want too many neurons with very high activation
    * this doesnt help that much.
    * no one really uses it

![alexnet_arch_extended](alexnet_arch_extended.png)

* AlexNet was the gamechanger -> 2012.
* One of the easier ones to read

### VGG-16

* instead of having massive amounts of parameters, use smaller network
* **always** use conv 3x3 filters, stride=1 , same padding
* maxpool **always** 2x2, s=2
* much deeper network
* Essentially, because SAME convolutions and max pooling, the network shrinks very easily readable
* output = softmax (1000)
* 16 in name VGG-16 refers to 16 layers that have weights
* 138mil params...
* number of filters are always doubled 64-128-256-512
* relative uniform network
* in literature, VGG-19 is a bigger version of VGG-16.
* vgg-16 almost does as well

`pattern: if you go deeper, H and W go down, but channels increase`

Architecture:

![vgg16_arch](vgg16_arch.png)

## ResNets: Residual Networks

* allows skip connections
* very deep > 100 layers

### Residual blocks

* $\overbrace{\ssb{a}{l} \underbrace{\rightarrow \textnormal{linear} \rightarrow \textnormal{ReLU} \rightarrow \ssb{a}{l+1} \rightarrow \textnormal{linear} \rightarrow}_{\textnormal{"shortcut / skip connection"}} \textnormal{ReLU} \rightarrow \ssb{a}{l+2}}^{\textnormal{main path}}$

$$\ssb{z}{l+1} = \ssb{W}{l+1}\ssb{a}{l} + \ssb{b}{l+1}$$
$$\ssb{a}{l+1} = g(\ssb{z}{l+1})$$
$$\ssb{z}{l+2} = \ssb{W}{l+2}\ssb{a}{l+1} + \ssb{b}{l+2}$$
$$\ssb{a}{l+2} = g(\ssb{z}{l+2})$$

* now, lets take $\ssb{a}{l}$ and make a "shortcut / skip connection". This makes that the output is

$$\ssb{a}{l+2} = g(\ssb{z}{l+2} + \ssb{a}{l})$$

![resnet_residual_blocks](resnet_residual_blocks.png)

* Using Residual Blocks allows to build alot DEEPER networks

### Residual "Plain Network"

* add all the skip connections to turn each of the blocks into residual blocks
* it turns out that if you use GD / ... As you increase the number of layers, the training error decreases but after a while it will go back up.
* in theory, having a deeper network should only help, but in reality the error goes back up
* helps with vanishing / explodin gradients

![resnet_plain_network](resnet_plain_network.png)

### Why ResNets work

* making bigger networks hurts the performance of the network
* this is not true if you use residual blocks

Lets start the idea with a big NN. A Different net will take the same network, but adds some extra layers in the form of a residual block

![resnet_whytheywork](resnet_whytheywork.png)

* Lets assume we use ReLU, so all activations are always greater than 0

$$\ssb{a}{l+2} = g(\ssb{z}{l+2} + \ssb{a}{l}) = g(\ssb{w}{l+2}\ssb{a}{l+1} + \ssb{b}{l+2} + \ssb{a}{l})$$

* if you apply weight decay or L2 regularization, that will tend to shrink the value of $\ssb{W}{l+2}$ (and b, but not as important). If this happens, the output becomes just equal to $\ssb{a}{l}$.
* the identity function is **easy** for residual blocks to learn because of the skip connection
* its difficult for normal networks to learn Identity, thats why larger networks hurt performance
* many times with residual blocks it will just help in performance

Some extra notes:

* Typically the SAME convolution is used, so the dimensions are also taken care of with the skip connection.
* if there is a difference in shape, we can add a new matrix to multiply so that the output becomes $W_s\ssb{a}{l}$ which is fo the same dimension.
* This matrix $W_s$ makes that the dimensions don't mismatch.
    * Can be a matrix with params to be learned
    * Can be a fixed matrix with zero padding: a_l with zero pads on the sides to make the dimensions work

![resnet_whytheywork_extra](resnet_whytheywork_extra.png)

### ResNet on images

* start from plain network
* add skip connections
* there's alot of 3x3 **same** convolutions, thats why we add equal dimension vectors (dimension is preserved)
* When there's a pooling layer, you need to make an adjustment to the dimension
* The last layer is a fully connected layer that makes a prediction using softmax

![resnet_whytheywork_example](resnet_whytheywork_example.png)

## Networks in Networks and 1x1 Convolutions

* 1x1 conv = multiplication is matmul with scalar
* However, in a high dimensional volume  (c>1), it makes more sense
    * look at each of the 32 different positions
    * takes elementwise product of 32numbers on the left and 32numbers of the filter.
    * after, it applies a relu
* think of it as one neuron that takes as input 32 numbers, multiplying them by 32 weights, applying nonlinearity and outputting
* if yuo have multiple convlutions, you have multiple units that take as input the slice and then building them up into a new output block

![networks_in_networks_and_1x1conv](networks_in_networks_and_1x1conv.png)

* its basically having a fully connected network that applies to each of the elements, so it takes as input 32 numbers and outputs a number of filters $\ssb{n}{l+1}_c$.
* can do difficult multiplication

![networks_in_networks_and_1x1conv_extra](networks_in_networks_and_1x1conv_extra.png)

* also named "Network in Network" , by lin et al.
* An example of where a 1by1 conv is useful
    * if the number of channels is too big and you want to shrink it (ie go from 128 -> 32 channels)
    * you can also keep the number of channels constant/decrease/increase
    * `can just use this as an extra nonlinearity`

![networks_in_networks_and_1x1conv_example](networks_in_networks_and_1x1conv_example.png)

## Inception Network

* arch is more complex, but works really well
* instead of choosing what size of convolution you want, do them all
* stack up multiple blocks of convolutions, making the dimensions match up
* also include pooling. In order to make the dimensions match, you need to add padding.

![inceptionnet_motivation](inceptionnet_motivation.png)

Above is called an inception volume

* actually increases the number of channels
* You dont need to pick any filter size / ...
* Let the network learn what it needs

However, the computational cost is very big. Lets focus on the 5x5 convolution of the above image below.

* you have 32 filters
* each filter is 5x5x192
* output is 28x28x32
* this means that we need to calculate 28x28x32 x 5x5x192 = 120million computations for only this block

![inceptionnet_motivation_5x5conv](inceptionnet_motivation_5x5conv.png)

Alternatively, we can use a 1x1 convolution to a 16 channel volume and consequently run the 5x5 convolution to get the final output.

* the input / output size is the same
* essentially shrink to `an intermediate volume called a bottleneck layer`
* the computational cost is now 
    * First 1x1 conv: 28x28x16 (outputs) * 1x1x192 (multiplications) = 2.4mil
    * second 5x5 conv: 28x28x32 (outputs) * 5x5x16 = 10.0mil
    * total amount of multiplications = 12.4mil
    * the number of additions is about the same as the amount of multiplications
    * reduction of ~90%!
* as long as you implement this bottleneck layer within reason, you can shrink this layer quite significantly without significantly reducing the performance.

![inceptionnet_motivation_bottleneck](inceptionnet_motivation_bottleneck.png)

## Building an Inception Network

* Combine all the blocks
* in the pooling layer, remember to use SAME padding for pooling, so that the output height/width can be concatenated with the other outputs. Even after this it will have the same amount of channels! Add one more 1x1 conv layer to shrink the number of channels.

![inceptionnet_arch](inceptionnet_arch.png)

This is one inception module. InceptionNet uses many of these modules

* There are some extra max pooling layers to change the height and width of the network.

![inceptionnet_arch_network](inceptionnet_arch_network.png)

One more detail: there are side branches (green). They take a hidden layer and use that to make a prediction (softmax). 

* helps to ensure that the computed features in intermediate layers are not too bad
* helps preventing overfitting
* also goes by the name GoogLeNet :)

![inceptionnet_arch_network_side_branch](inceptionnet_arch_network_side_branch.png)

* inceptionNet actually refers this meme below, as motivation for the need to build deeper NN

![inceptionnet_meme](inceptionnet_meme.png)

