import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)


# find the best combination of  hyperparameters
# use GridSearchCV or RandomizedSearchCV

def build_model(n_hidden =1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    model = keras.models.Sequential()
    options = {"input_shape": input_shape}
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation='relu', **options))
        options = {}
    model.add(keras.layers.Dense(1, **options))
    optimizer = keras.optimizers.SGD(learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

# a thin wrapper around the Keras model
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
# Now we can use this object like a regular Scikit-Learn regressor
# TODO: what's the difference between this function and the Wide and Deep neural network defined before?
# we can train it using its fit() method, 
# then evaluate it using its score() method, 
# and use it to make predictions using its predict() method.

keras_reg.fit(X_train_scaled, y_train, epochs=100,
                validation_data = (X_valid_scaled, y_valid),
                callbacks=[keras.callbacks.EarlyStopping(patience=10)])
mse_test = keras_reg.score(X_test, y_test)
# note that the score will be the opposite of the MSE 
# because Scikit-Learn wants scores, not losses (i.e., higher should be better).
y_pred = keras_reg.predict(X_train_scaled[:3])

# use randomized search to explore the number of hidden layers, the number of neurons and the learning rate
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    "n_hidden": [0, 1, 2, 3],
    "n_neurons": np.arange(1, 100),
    "learning_rate": reciprocal(3e-4, 3e-2),
}
rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
rnd_search_cv.fit(X_train, y_train, epochs=100,
                    validation_data = (X_valid, y_valid),
                    callbacks=[keras.callbacks.EarlyStopping(patience=10)])

best_params = rnd_search_cv.best_params_
best_score = rnd_search_cv.best_score_
model = rnd_search_cv.best_estimator_.model

# optimize the hyperparameters efficiently
# zoom in on the region of the space where the hyperparameters performances are good
# Hyperopt, Hyperas, kopt , Talos, Scikitt-Optimize, Spearmint, Sklearn-Deap

# hyperparameter tuning algorithms
# “Population Based Training of Neural Networks,” Max Jaderberg et al. (2017).

### guidelines for choosing the number of hidden layers and neurons in an MLP, and selecting good values for some of the main hyperparameters.
## number of hidden layers
# deep networks have a much higher parameter efficiency than shallow ones: 
# they can model complex functions using exponentially fewer neurons than shallow nets, 
# allowing them to reach much better performance with the same amount of training data.

# it also improves their ability to generalize to new datasets.
# you can initialize them to the value of the weights and biases of the lower layers of the first network.
# this is called transfer learning.

## number of neurons per hidden layer
# paramid structure has been largely abandoned now
# depending on the dataset, it can sometimes help to make the first hidden layer bigger than the others.
# A simpler approach is to pick a model with more layers and neurons than you actually need, 
# then use early stopping or other refularization techniques to prevent it from overfitting

## Learning Rate, Batch Size and Other Hyperparameters
# learning rate
# maximum learning rate: the learning rate above which the training algorithm diverges
# optimal learning rate is about half of the maximum 
# a simple approach for tuning the learning rate is to start with a large value that makes the training algorithm diverge, 
# then divide this value by 3 and try again, and repeat until the training algorithm stops diverging.

# optimizer
# In general the optimal batch size will be lower than 32

# activation function
# in gen‐ eral, the ReLU activation function will be a good default for all hidden layers. 
# For the output layer, it really depends on your task.

# using early stopping for the number of training iterations

# “Practical recommendations for gradient-based training of deep architectures,” Yoshua Bengio (2012).
