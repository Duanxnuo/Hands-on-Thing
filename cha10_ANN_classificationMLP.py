import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

##### Classification MLP #####
## Building an Image Classifier Using the Sequential API
#load the fashion_mnist dataset
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
# print(X_train_full.shape)
# print(X_train_full.dtype)
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
class_names = ['T-shirt/top',  "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# print(class_names[y_train[0]])
model = keras.models.Sequential() # create a sequential model for neural networks that are just composed of a single stack of layers, connected sequentially
model.add(keras.layers.Flatten(input_shape=[28,28])) # computes input X.reshape(-1, 1), can be alternatively model.add(keras.layers.InputLayer(shape=[28,28]))
model.add(keras.layers.Dense(300, activation='relu')) # 300 neurons
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax')) # 10 neurons, one per class

'''
model = keras.model.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(300, activation = 'relu'),
    keras.layers.Dense(300, activation = 'relu'),
    keras.layers.Dense(10, activation='softmax')
])
'''

# print(model.summary())
# model.layers[1].name
# model.get_layer('dense')
# all the parameters of a layer can be accessed using its get_weights() and set_weights() methods
# weights, biases = hidden1.get_weights()
# dense layer initialized the connection weights randomly and the biases were just initialized to zeros
# use kernel_initializer and bias_initializer to customize the initialization method when creating the layer

## compiling the model
# specify the loss function and the optimizer to use
model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])
# sparse because we have sparse labels
# is we deal with one-hot vectors, we would need to use the "categorical_crossentropy" loss
# binary classification use sigmoid activation function (logistic) binary_crossentropy loss
# np.argmax()
# softmax activation function for probability output

# because this model is a classifier, it's useful to measure its accuracy during training and evaluation

# loss = sparse_cateforical_crossentropy
# loss = keras.losses.sparse_categorical_crossedntropy
# optimizer = "sgd"
# optimizer = keras.optimizers.SGD()
# metrics=['accuracy']
# metrics = [keras.metrics.sparse_categorical_accuracy]

# convert sparse labels to one-hot vector labels: keras.utils.to_categorical()
# convert one-hot to sparse labels np.argmax() with axis = 1

## training and evaluating the model
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

## plot the metrics
# fit returns a History object containing the training parameters(history.params)
# (history.epoch) the list of epochs it went through
# (history.history) containing the loss the extra metrics

# pd.DataFrame(history.history).plot(figsize=(8,5))
# plt.grid(True)
# plt.gca().set_ylim(0,1) # set the vertical range to [0-1]
# plt.show()

# Once you are satisfied with your model’s validation accuracy, 
# you should evaluate it on the test set to estimate the generalization error before you deploy the model to production.

model.evaluate(X_test/255, y_test)

## Using the Model to Make Predictions
X_new = X_test[:3] / 255
y_proba = model.predict(X_new)
y_proba.round(2)

