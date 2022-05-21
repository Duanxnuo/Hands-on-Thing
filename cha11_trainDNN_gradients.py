from tensorflow import keras
#########################################
# Vanishing/Exploding Gradients Problem #
#########################################
# appear in backpropagation steps
# gradients often get smaller and smaller as the algorithm progresses down to the lower layers.
# As a result, the Gradient Descent update leaves the lower layer connection weights virtually unchanged, 
# and training never converges to a good solution. 
# sometimes the opposite would happen, which is called exploding gradients, 
# often encountered in recurrent neural networks
# In general,
# deep neural networks suffer from unstable gradients; different layers may learn at widely different speeds.

# “Understanding the Difficulty of Training Deep Feedforward Neural Networks,” X. Glorot, Y Bengio (2010).
# Looking at the logistic activation function, 
# you can see that when inputs become large (negative or positive), 
# the function saturates at 0 or 1, with a derivative extremely close to 0. 
# Thus when backpropagation kicks in, it has virtually no gradient to propagate back through the network,
# and what little gradient exists keeps getting diluted as backpropagation progresses down through the top layers, 
# so there is really nothing left for the lower layers.

####### Glorot and He Initialization #######
# we need the variance of the outputs of each layer to be equal to the variance of its inputs
# and we also need the gradients to have equal variance before and after flowing through a layer in the reverse direction
# It is actually not possible to guarantee both unless the layer has an equal number of inputs and neurons
# which are fan-in and fan-out numbers
# a compromised way to solve these:
# the connection weights of each layer must be initialized randomly
# By default, Keras uses Glorot initialization with a uniform distribution.

keras.layers.Dense(10, activation="relu", kernel_initializer="he_normal")
he_avg_init = keras.initializers.VarianceScaling(scale=2., mode='fan_avg',
                                                     distribution='uniform')
keras.layers.Dense(10, activation="sigmoid", kernel_initializer=he_avg_init)

####### Nonsaturating Activation Functions ######
# the vanishing/ exploding gradients problems were in part due to a poor choice of activation function.
# ReLu does not saturate for positive values and also is fast to compute
# however it suffers from a problem known as the dying ReLUs
# during training, some neurons effectively die, meaning they stop outputting anything other than 0
# use leaky ReLU
# LeakyReLU_{a}(z) = max(az, z) a is the slope of the function and is typically set to 0.01
# ensures that leaky ReLUs never die
# ELU outperformed all the ReLU variants in their experiments
# “Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs),” D. Clevert, T. Unterthiner, S. Hochreiter (2015).

Leaky_relu = keras.layers.LeakyReLU(alpha = 0.2)
layer = keras.layers.Dense(10, activation = "leaky_relu",
                            kernel_initializer='he_normal')

layer = keras.layers.Dense(10, activation="selu",
                            kernel_initializer="lecun_normal")

###### Batch Normalization ######
# scaling parameter and shifting parameter
# this operation lets the model learn the optimal scale and mean of each of the layer’s inputs.
# In order to zero-center and normalize the inputs, 
# the algorithm needs to estimate each input’s mean and standard deviation. 
# It does so by evaluating the mean and standard deviation of each input over the current mini-batch 
# (hence the name “Batch Normalization”).

# implementing 
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, activation = 'elu', kernal_initializer='he_normal'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation = 'elu', kernel_initializer='he_normal'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation='softmax')
])

# [(var.name, var.trainable) for var in model.layers[1].variables]
# model.layers[1].updates
# since a Batch Normaliza‐ tion layer includes one offset parameter per input, you can remove the bias term from the previous layer

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, kernel_initializer='he_normal', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('elu'),
    keras.layers.Dense(100, kernel_initializer="he_normal", use_bias=False),
    keras.layers.Activation("elu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation="softmax")
])

# parameters for batchnormalization: 
# momentum
# This hyperparameter is used when updating the exponential moving averages: 
# given a new vector of input means or standard deviations computed over the current batch
# the running average paramter v1 is updated using the following equation
# v1 <- v1 * momentum + v *(1-momentum)
# A good momentum value is typically close to 1
# axis
# determinnes which axis should be normalized, defaults to -1
# meaning the default is to normalize the last axis
# using the means and standard deviations computed across the other axes
# BN layer uses batch statistics during training and the final statistics after training
# Batch Normalization has become one of the most used layers in deep neural net‐ works, 
# to the point that it is often omitted in the diagrams, 
# as it is assumed that BN is added after every layer. 
# “Fixup Initialization: Residual Learning Without Normalization,” Hongyi Zhang, Yann N. Dauphin, Tengyu Ma (2019).
# by using a novel fixed-update (fixup) weight initialization technique, they manage to train a very deep neural network (10,000 layers!) without BN

###### Gradient Clipping #####
# Another popular technique to lessen the exploding gradients problem
# most often used in recurrent neural networks, as Batch Normalization is tricky to use in RNNs
optimizer = keras.optimizers.SGD(clipvalue = 1)
model.compile(loss='mse', optimizer = optimizer)
# This will clip every component of the gradient vector to a value between –1.0 and 1.0.
# when setting the clipvalue, it may change the orienntation of the gradient vector
# to ensure that Gradient Clipping does not change the direction of the vector
# use clipnore instead of clipvalue
# This will clip the whole gradient if its l2 norm is greater than the threshold you picked.
# if observe gradient exploding during training, you can try both clippin gby value and clipping by norm


