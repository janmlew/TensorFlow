"""

My project on DeepLearning.AI TensorFlow Developer Professional Certificate.

Course 1., week 2.

"""

# Imports:
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Data loads:


# Load the Fashion MNIST dataset
fmnist = tf.keras.datasets.fashion_mnist
# Load the training and test split of the Fashion MNIST dataset
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()


# Main code:
# Build the classification model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

