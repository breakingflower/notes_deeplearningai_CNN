# week 2 quiz

## q1 

Which of the following do you typically see as you move to deeper layers in a ConvNet?

* [ ] $n_H, n_W$​ increases, while $n_C$ also increases

* [ ] $n_H, n_W$ decreases, while $n_C$​ also decreases

* [ ] $n_H, n_W$ increases, while $n_C$​ decreases

* [x] $n_H, n_W$ decrease, while $n_C$​ increases `width height decrease, channels increase`

## q2

Which of the following do you typically see in a ConvNet? (Check all that apply.)

* [x] Multiple CONV layers followed by a POOL layer

* [ ] Multiple POOL layers followed by a CONV layer

* [x] FC layers in the last few layers

* [ ] FC layers in the first few layers

## q3 

In order to be able to build very deep networks, we usually only use pooling layers to downsize the height/width of the activation volumes while convolutions are used with “valid” padding. Otherwise, we would downsize the input of the model too quickly.

* [] True `incorrect!` 

* [x] False `it shuold be false`

## q4

Training a deeper network (for example, adding additional layers to the network) allows the network to fit more complex functions and thus almost always results in lower training error. For this question, assume we’re referring to “plain” networks.

* [ ] True

* [x] False `actually the opposite, thats why residual networks are commonly used`

## q5

The following equation captures the computation in a ResNet block. What goes into the two blanks above?

$$ \ssb{a}{l+2} = g(\ssb{W}{l+2}g(\ssb{W}{l+1}\ssb{a}{l}+\ssb{b}{l+1})+\ssb{b}{l+2} + ...) + ....$$

`the second g term is the activations of layer 1, so the residual block is still missing the `$\ssb{a}{l}$ `term. The second one is 0 because there is nothing after this g function`

## q6 


Which ones of the following statements on Residual Networks are true? (Check all that apply.)

`some errors here`

* [x] Using a skip-connection helps the gradient to backpropagate and thus helps you to train deeper networks. `the fact you can learn identity helps pass gradients that might disappear without skipconnections`

* [ ] A ResNet with L layers would have on the order of $L^2$ skip connections in total. `no L/2`

* [ ] The skip-connections compute a complex non-linear function of the input to pass to a deeper layer in the network. `i mean yes, thats just what conv layers do (with relu).. but no`

* [x] The skip-connection makes it easy for the network to learn an identity mapping between the input and the output within the ResNet block. 

## q7 

Suppose you have an input volume of dimension 64x64x16. How many parameters would a single 1x1 convolutional filter have (including the bias)?

`a 1x1 conv takes a slice of over the channel dimension of the volume, so this is 16. if yuo add the bias term this becomes 17`

## q8

Suppose you have an input volume of dimension $n_H, n_W, n_C$. Which of the following statements you agree with? (Assume that “1x1 convolutional layer” below always uses a stride of 1 and no padding.)

* [x] You can use a pooling layer to reduce  $n_H, n_W$ but not $n_C$​.

* [ ] You can use a 1x1 convolutional layer to reduce  $n_H, n_W, n_C$

* [x] You can use a 1x1 convolutional layer to reduce $n_C$ but not $n_H, n_W$.

* [ ] You can use a pooling layer to reduce $n_H, n_W, n_C$

## q9

Which ones of the following statements on Inception Networks are true? (Check all that apply.)
`some errors here`

* [ ] Making an inception network deeper (by stacking more inception blocks together) should not hurt training set performance. `will actually hurt as this boils down to why there are residual networks in the first place (vanishing gradients /..). Maybe a residual inception block can help overcome this`

* [ ] Inception networks incorporates a variety of network architectures (similar to dropout, which randomly chooses a network architecture on each step) and thus has a similar regularizing effect as dropout. `dropout disables neurons, inception adds extra conv layers in parallel`

* [x] Inception blocks usually use 1x1 convolutions to reduce the input data volume’s size before applying 3x3 and 5x5 convolutions. `yess`

* [x] A single inception block allows the network to use a combination of 1x1, 3x3, 5x5 convolutions and pooling. `yes`

## q10

Which of the following are common reasons for using open-source implementations of ConvNets (both the model and/or weights)? Check all that apply.

* [x] It is a convenient way to get working an implementation of a complex ConvNet architecture.

* [ ] A model trained for one computer vision task can usually be used to perform data augmentation even for a different computer vision task.

* [ ] The same techniques for winning computer vision competitions, such as using multiple crops at test time, are widely used in practical deployments (or production system deployments) of ConvNets.

* [x] Parameters trained for one computer vision task are often useful as pretraining for other computer vision tasks.
