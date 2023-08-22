"""

My project on DeepLearning.AI TensorFlow Developer Professional Certificate.

Main project file.

"""

# Imports:
import numpy as np
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)


# Data loads:


# Main code:


# Non-neural equivalent of the 'hello world' model:

def hw_function(x):
    y = (x * 2) - 1
    return y


# The 'hello world!' example ;)

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))
