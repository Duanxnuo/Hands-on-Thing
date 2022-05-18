import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

##### Regression MLP #####
# output layer has a single neuron
# use no activation function
# loss function is mse
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])
model.compile(loss="mean_squared_error", optimizer="sgd")
history = model.fit(X_train, y_train, epochs = 20, validation_data=(X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)

##### Building Complex Models Using the Functional API
# build neural networks with complex topolocies or with multiple inputs or outputs
# non-sequential neural network
# Wide & Deep neural network HEng-Tze Cheng
# connects all or part of the inputs directly to the output layer
# This architecture makes it possible for the neural network to learn 
# both deep patterns (using the deep path) and simple rules (through the short path).

# In contrast, a regular MLP forces all the data to flow through the full stack of layers, 
# thus simple patterns in the data may end up being distorted by this sequence of transformations.

input = keras.layers.Input(shape=X_train.shape[1:])
# telling Keras how it should connect the layers together, no actual data is being processed yet
hidden1 = keras.layers.Dense(30, activation="relu")(input) # As soon as it is created, notice that we call it like a function, passing it the input. This is why this is called the Functional API.
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.Concatenate()([input, hidden2]) # to concatenate the input and the output of the second hidden layer
# single neuron and no activation function
output = keras.layers.Dense(1)(concat)
model = keras.models.Model(inputs=[input], outputs=[output])

# send a subset of the features through the wide path, 
# and a different subset (possibly overlapping) through the deep path
# use multiple inputs
input_A = keras.layers.Input(shape=[5])
input_B = keras.layers.Input(shape=[6])
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.models.Model(inputs=[input_A, input_B], outputs=[output])

model.compile(loss="mse", optimizer="sgd")

X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

history = model.fit((X_train_A, X_train_B), y_train, epochs=20,
                    validation_data=((X_valid_A, X_valid_B), y_valid))
mse_test = model.evaluate((X_test_A, X_test_B), y_test)
y_pred = model.predict((X_new_A, X_new_B))

output = keras.layers.Dense(1)(concat)
aux_output = keras.layers.Dense(1)(hidden2)
model = keras.models.Model(inputs = [input_A, input_B], 
                            outputs = [output, aux_output])
model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1], optimizer="sgd")
history = model.fit(
        [X_train_A, X_train_B], [y_train, y_train], epochs=20,
        validation_data=([X_valid_A, X_valid_B], [y_valid, y_valid]))
total_loss, main_loss, aux_loss = model.evaluate(
        [X_test_A, X_test_B], [y_test, y_test])
y_pred_main, y_pred_aux = model.predict([X_new_A, X_new_B])


##### Building Dynamic Models Using the Subclassing API
# Some models involve loops, varying shapes, conditional branching, and other dynamic behaviors. 
# imperative programming style
# Keras models can be used just like regular layers,
# so you can easily compose them to build complex architectures.
class WideAndDeepModel(keras.models.Model):
    def __init__(self, units = 30, activation = "relu", **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(units, activation = activation)
        self.hidden2 = keras.layers.Dense(units, ativation = activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1) # combine the same functions?

    # customized call functions
    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output
    
model = WideAndDeepModel()
# dont have to create the inputs 
# TODO: why? why dont specify the shape of the inputs?
# process the inputs before calling the call method
# flip side:
# architecture is hidden within the call() method, so Keras cannot easily inspect it, 
# it cannot save or clone it, and when you call the summary() method, 
# you only get a list of layers, without any information on how they are connected to each other. 
# and keras cannot check types and shapes ahead of time 
# so it's easier to make mistakes

##### saving and restoring a model
model.save("my_keras_model.h5")
# save both the model's architecture including layers' hyperparameters
# and the value of all the model parameters for each layer
# connection weights and biases
# optimizer
# using HDF5 format
model = keras.models.load_model("my_keras_model.h5")
# not working when subclassing
# store every parameters and metrics manually 
# save_weights() and load_weights()

##### using callbacks to save checkpoints
checkpoint_cb = keras.callbacks.Modelcheckpoint("my_keras_model.h5")
history = model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint_cb])
# implement early stopping by saving the best performance model on validation set
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5",
                                                    save_best_only=True)
history = model.fit(X_train, y_train, epochs=10,
                        validation_data=(X_valid, y_valid),
                        callbacks=[checkpoint_cb])
# restore the last model saved after training
model = keras.models.load_model("my_keras_model.h5") # roll back to best model

## use EarlyStopping callback
# can combine both callbacks to both save checkpoints of your model
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True) 
# no need to restore the best model
# since the EarlyStopping callback will keep track of the best weights and restore them for us at the end of training.
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks = [checkpoint_cb, early_stopping_cb])

# custom callback
class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))
# TODO: what's the logs?
# callbacks can be called in fit(), evaluate(), predict()

##### Visualization Using TensorBoard
root_logdir = os.path.join(os.curdir, "my_logs")
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit(X_train, y_train, epochs=30, validation_data = (X_valid, y_valid), callbacks = [tensorboard_cb])

