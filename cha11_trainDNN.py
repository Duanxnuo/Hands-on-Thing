from tensorflow import keras

#######################################
###### Reusing Pretrained Layers ######
#######################################
# you should always try to find an existing neural network that accomplishes a similar task to the one you are trying to tackle
# Try freezing all the reused layers first (i.e., make their weights non-trainable, so gradi‐ ent descent won’t modify them)
# then train your model and see how it performs
# then try unfreezing one or two of the top hidden layers to let backpropagation tweak them and see if oerformance improves
# It is also useful to reduce the learning rate when you unfreeze reused layers: this will avoid wrecking their fine-tuned weights.
# If you have plenty of train‐ ing data, you may try replacing the top hidden layers instead of dropping them, and even add more hidden layers.

## Transfer Learning with Keras
# reuse all layers except for the output layer
model_A = keras.models.load_model('my_model_A.h5')
model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
model_B_on_A.add(keras.layers.Dense(1, activation='sigmoid'))
# Note that model_A and model_B_on_A now share some layers. When you train model_B_on_A, it will also affect model_A. 
# If you want to avoid that, you need to clone model_A before you reuse its layers.
# clone model and copy its weights
model_A_clone = keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())
# freeze the reused layers for the first few epochs
# because the new output layer was initialized randomly, it will make large errors
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False
model_B_on_A.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
# You must always compile your model after you freeze or unfreeze layers.
# After unfreezing the reused layers, it is usually a good idea to reduce the learning rate, once again to avoid damaging the reused weights
history = model_B_on_A.fit(X_train_B, y_train_B, epochs = 4,
                            validation_data=(X_valid_B, y_valid_B))
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True

optimizer = keras.optimizer.SGD(lr = 1e-4) # the default is 1e-3
model_B_on_A.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16, validation_data=(X_valid_B, y_valid_B))

# model_B_on_A.evaluate(X_test_B, y_test_B)
# transfer learning does not work very well with small dense networks: 
# it works best with deep convolutional neural networks

##### unsupervised pretraining #####
# less labeled data
# If you can gather plenty of unlabeled training data, you can try to train the layers one by one, 
# starting with the lowest layer and then going up, 
# using an unsupervised feature detector algorithm 
# such as Restricted Boltzmann Machines (RBM) or autoencoders
# Each layer is trained on the output of the previously trained layers 
# (all layers except the one being trained are frozen)
# Once all layers have been trained this way, you can add the output layer for your task, and fine-tune the final network using supervised learning (i.e., with the labeled training examples). 
# At this point, you can unfreeze all the pretrained layers, or just some of the upper ones.

##### Pretraining on an Auxiliary Task #####
# If you do not have much labeled training data, 
# one last option is to train a first neural network on an auxiliary task for which you can easily obtain or generate labeled training data, 
# then reuse the lower layers of that network for your actual task. 

##### faster optimizers #####
## previous introduced ways to speed up training and reach a better solution ##
# applying a good initialization strategy for the connection weights
# using a good activation function
# using Batch Normalization
# reusing parts of a pretrained network

# most popular optimizers:
# Momentum optimization
# Nestero Accelerated Gradient
# AdaGrad
# PMSProp
# Adam and Nadam optimization

### Momentum Optimization ###
# regular gradient descent does not care about what the earlier gradients were
# if the local gradient is tiny, it goes very slowly
# momentum optimization cares a great deal about what previous gradients were
# at each iteration, it subtracts the local gradient from the momentum vector (multiplied by the learning rate)
# and update weights by simplyadding this momentum vector
# momentum vector
# learning rate
# momentum
# gradient
# momentum vector <- momentum * momentum vector - learning rate * gradient of the cost function with respect to the weights
# new weights <- weights + momentum vector
# the gradient is used for acceleration not for speed
# the greater the momentum is, the momentum optimization going faster
# this allows momentum optimization to escape from plateaus much faster than Gradient Descent
optimizer = keras.optimizers.SGD(lr = 0.001, momentum=0.9)
# However, the momentum value of 0.9 usually works well in practice and almost always goes faster than regular Gradient Descent.

### Nesterov Accelerated Gradient
# a little modifier to the vanilla Momentum optimization
# to measure the gradient of the cost function not at the local position but slightly ahead in the direction of the momentum
# TODO: not really under stand???
# momentum vector <- momentum * momentum vector - learning rate * (gradient of the cost function with respect to the weights + momentum * momentum vector)
# new weights <- weights + momentum vector
# This small tweak works because in general the momentum vector will be pointing to the right direction
# so it will be slightly more accurate to use the gradient measured a bit farther in that direction rather than using the gradi‐ ent at the original position\
optimizer = keras.optimizers.SGD(lr = 0.001, momentum=0.9, nesterov=True)


#### AdaGrad
# Gradient Descent stars by quickly going down the steepest slope, 
# then slowly goes down the bottom of the valley
# it would be nice if the algorithm could detect this early on 
# and correct its direction to point a bit more toward the global optimum
# the adagrad algorithm achieves this by scaling down the gradient vector along the steepest dimensions
# helps point the resulting updates more directly toward the global optimum
# requires much less tuning of the learning rate hyperparameter
# AdaGrad often performs well for simple quadratic porblems, 
# but often stops too early when training neural networks
# the learch rate gets scaled down so much that the algorithm ends up stopping entirely before reaching the global optimum

#### RMSProp
# RMSProp fix the AdaGrad slowing down a bit too fast and ending up never convergint to the global optimum
# by accumulating only the gradients from the most recent iterations
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9)


#### Adam and Nadam OPtimization
# adaptive moment estimation, combines the ideas of Momentum optimization and RMSProp
# keeps track of an exponentially decaying average of past gradients
# keeps track of an exponentially decaying average of past squared gradients
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.9)
## two variants from Adam
# Adamax
# Nadam optimization
 
##### Learning Rate Scheduling #####
# reduce the learning rate during the training once it stios making fast progress
## Power scheduling 
optimizer = keras.optimzers.SGD(lr=0.01, decay=1e-4)
## exponential scheduling
## piecewise constant scheduling 
# 1.define a function that takes the current epoch and returns the learning rate
def exponential_decay_fn(epoch):
    return 0.012 * 0.1 ** (epoch/20)
# or create a function that returns a configured function
def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0* 0.1 **(epoch/s)
    return exponential_decay_fn
exponential_decay_fn = exponential_decay_fn(lr = 0.01, s = 20)
# 2. create a LearningRateScheduler callback
lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
history = model.fit(X_train_scaled, y_train, [...], callbacks=[lr_scheduler])

## Performance scheduling
# multiply the learning rate by 0.5 whenever the best validation loss does not improve for 5 consecutive epochs
lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
 
# alternative ways offered by tf.keras
s = 20*len(X_train) // 32 # number of steps in 20 epochs, batch size = 32
learning_rate = keras.optimizers.schedules.ExponentialDecay(0.01, s, 0.1)
optimizer = keras.optimizers.SGD(learning_rate)


##### Avoiding Overfitting Through Regularization #####
# use l1, l2 regularization to constrain a neural network's connection weights(but typically not its biases) 
# using a regul arization factor of 0.01
layer = keras.layers.Dense(100, activation='elu',
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(0.01))

# use python functols.partial() function to create a thin wrapper for any callable
# to avoid repeating the same arguments over and over
from functools import partial
RegularizedDense = partial(keras.layers.Dense, activation='elu', kernel_initializer = 'he_normal',
                            kernel_regularizer=keras.regularizers.l2(0.01))
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    RegularizedDense(300),
    RegularizedDense(100),
    RegularizedDense(10, activation='softmax',
                        kernel_initializer='glorot_uniform')
])                            


##### Dropout ######
# at every training step, every neuron has a probability p of being temporarily "dropped out",
# meaning it will be entirely ignored during this training step,
# but it may be active during the next step
# the hyperparameter p is called the dropout rate and typically set to 50% 
# “Improving neural networks by preventing co-adaptation of feature detectors,” G. Hinton et al. (2012).
# “Dropout: A Simple Way to Prevent Neural Networks from Overfitting,” N. Srivastava et al. (2014).
# Neurons trained with dropout cannot co-adapt with their neighboring neurons, 
# they have to be as useful as possible on their own
# they also cannot rely excessively on just a few inpu neurons, they must pay attention to each of their input neurons
# they end up being less sensitive to slight changes in the inputs
# a unique neural network is generated at each training step
#  they are not independent 
# the resulting neural network can be seen as an averaging ensemble of all these smaller neural networks
# technical detail::
# suppose p = 50%, in which case during testing a neuron will be connected to twice as many input neurons as it was on average during training 
# to compensate for this fact, we need to multiply each neuron's input connection weights by 0.5 after training
# More generally, we need to multiply each input connec‐ tion weight by the keep probability (1 – p) after training.
# Alternatively, we can divide each neuron’s output by the keep probability during training
# keras.layers.Dropout layer randomly drops some inputs and divides the remaining inputs by the keep probability
model = keras.models.Sequential([
    keras.layers.Faltten(input_shape=[28,28]),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(300, activation='elu', kernel_initializer='he_normal'),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    keras.layers.Dropour(rate=0.2),
    keras.layers.Dense(10, activation='softmax')   
])
# Since dropout is only active during training, the training loss is penalized compared to the validation loss, so comparing the two can be misleading.
# So make sure to evaluate the training loss without dropout (e.g., after training).
# Alternatively, you can call the fit() method inside a with keras.backend.learning_phase_scope(1) block: this will force dropout to be active during both training and validation.
# and set it back to 0 right after
# If you want to regularize a self-normalizing network based on the SELU activation function (as discussed earlier), 
# you should use AlphaDropout: 
# this is a variant of dropout that preserves the mean and standard deviation of its inputs (it was introduced in the same paper as SELU, as regular dropout would break self-normalization).

### Monte-Carlo Dropout
with keras.backend.learning_phase_scope(1): # force treainingmode=dropout on
    y_probas = np.stack([model.predict(X_test_scales) for sample in range(100)])
y_proba = y_probas.mean(axis=0)

# np.round(model.predict(X_test_scaled[:1]), 2)
# np.round(y_probas[:, :1], 2)
# np.round(y_proba[:1], 2)
# If your model contains other layers that behave in a special way during training
# such as Batch Normalization layers, then you should not force training mode like we just did
# Instead you should repalce the Dropout layers with the following MCDropout class
class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training = True)
# Similarly, you could define an MCAlphaDrop out class by subclassing AlphaDropout instead.
# But if you have a model that was already trained using Dropout, you need to create a new model, iden‐ tical to the existing model except replacing the Dropout layers with MCDropout, then copy the existing model’s weights to your new model.


### Max-Norm Regularization
# for each neuron it constrains the weights w of the incoming connections such that
# the l2 norm of the weights is less or equal to the max-norm hyperparameter
keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal', kernel_constraint=keras.constraints.max_norm(1.))
# After each training iteration, the model’s fit() method will call the object returned by max_norm(), passing it the layer’s weights and getting clipped weights in return, which then replace the layer’s weights. 


##### Practical GUidelines #####
# kernel initializer: LeCun initialization
# Activation function: SELU
# NOrmalization: None(self-normalization)
# Regularization: Early Stopping
# Optimizer: Nadam
# Learning rate schedule: Performance schedualing
# standardize teh input features


