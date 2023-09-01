"""

My project on DeepLearning.AI TensorFlow Developer Professional Certificate.

Course 1., week 2.

"""

# Imports:
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the Fashion MNIST dataset
fmnist = tf.keras.datasets.fashion_mnist
# Load the training and test split of the Fashion MNIST dataset
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

# There are between 0 and 59999 images available in the training set. Choose one in the index variable below:
index = 0

# Set number of characters per row when printing
np.set_printoptions(linewidth=320)

# Print the label and image
print(f'LABEL: {training_labels[index]}')
print(f'\nIMAGE PIXEL ARRAY:\n {training_images[index]}')

# Visualize the image
plt.imshow(training_images[index])
plt.show()

# Normalize the pixel values of the train and test images
training_images = training_images / 255.0
test_images = test_images / 255.0

# Build the classification model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

"""

Softmax takes a list of values and scales these so the sum of all elements will be equal to 1. When applied to model 
outputs, you can think of the scaled values as the probability for that class. For example, in your classification 
model which has 10 units in the output dense layer, having the highest value at index = 4 means that the model is 
most confident that the input clothing image is a coat. If it is at index = 5, then it is a sandal, and so forth. See 
the short code block below which demonstrates these concepts. You can also watch this lecture if you want to know 
more about the Softmax function and how the values are computed.

"""

# Declare sample inputs and convert to a tensor
inputs = np.array([[1.0, 3.0, 4.0, 2.0]])
inputs = tf.convert_to_tensor(inputs)
print(f'input to softmax function: {inputs.numpy()}')

# Feed the inputs to a softmax activation function
outputs = tf.keras.activations.softmax(inputs)
print(f'output of softmax function: {outputs.numpy()}')

# Get the sum of all values after the softmax
sum_after_softmax = tf.reduce_sum(outputs)
print(f'sum of outputs: {sum_after_softmax}')

# Get the index with the highest value
prediction = np.argmax(outputs)
print(f'class with highest probability: {prediction}')

"""

That's the example. Now let's get back to the actual model generation.

"""

model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

# Evaluate the model on unseen data
print(model.evaluate(test_images, test_labels))
