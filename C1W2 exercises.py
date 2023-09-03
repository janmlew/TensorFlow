"""

My project on DeepLearning.AI TensorFlow Developer Professional Certificate.

Course 1., week 2. exercises.

"""

# Imports:
import tensorflow as tf

# Model:
fmnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

# Exercises:

"""
Exercise 1. For this first exercise run the below code: It creates a set of classifications for each of the test 
images, and then prints the first entry in the classifications. The output, after you run it is a list of numbers. Why 
do you think this is, and what do those numbers represent?
"""

classifications = model.predict(test_images)
print(classifications[0])

# Hint: try running print(test_labels[0]) -- and you'll get a 9. Does that help you understand why this list looks
# the way it does?

print(test_labels[0])

"""
E1Q1: What does this list represent?
It's 10 random meaningless values
It's the first 10 classifications that the computer made
It's the probability that this item is each of the 10 classes
Click for Answer
Answer:
The correct answer is (3)

The output of the model is a list of 10 numbers. These numbers are a probability that the value being classified is 
the corresponding value (https://github.com/zalandoresearch/fashion-mnist#labels), i.e. the first value in the list 
is the probability that the image is of a '0' (T-shirt/top), the next is a '1' (Trouser) etc. Notice that they are 
all VERY LOW probabilities.

For index 9 (Ankle boot), the probability was in the 90's, i.e. the neural network is telling us that the image is 
most likely an ankle boot.
"""

"""
Exercise 2: Let's now look at the layers in your model. Experiment with different values for the dense layer with 
512 neurons. What different results do you get for loss, training time etc? Why do you think that's the case?
"""

fmnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    # Try experimenting with this layer
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy')

model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])

"""
E2Q1: Increase to 1024 Neurons -- What's the impact?
1) Training takes longer, but is more accurate
2) Training takes longer, but no impact on accuracy
3) Training takes the same time, but is more accurate
Click for Answer Answer The correct answer is (1) by adding more Neurons we have to do more calculations, slowing down 
the process, but in this case they have a good impact -- we do get more accurate. That doesn't mean it's always a case 
of 'more is better', you can hit the law of diminishing returns very quickly!
"""

"""
Exercise 3:
E3Q1: What would happen if you remove the Flatten() layer. Why do you think that's the case?Click for Answer
Answer
You get an error about the shape of the data. It may seem vague right now, but it reinforces the rule of thumb that the 
first layer in your network should be the same shape as your data. Right now our data is 28x28 images, and 28 layers of 
28 neurons would be infeasible, so it makes more sense to 'flatten' that 28,28 into a 784x1. Instead of writing all the 
code to handle that ourselves, we add the Flatten() layer at the begining, and when the arrays are loaded into the model
 later, they'll automatically be flattened for us.
"""

fmnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),  # Try removing this layer
                                    tf.keras.layers.Dense(64, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy')

model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])

"""
Exercise 4:
Consider the final (output) layers. Why are there 10 of them? What would happen if you had a different amount than 10? 
For example, try training the network with 5.

Click for Answer

Answer
You get an error as soon as it finds an unexpected value. Another rule of thumb -- the number of neurons in the last 
layer should match the number of classes you are classifying for. In this case it's the digits 0-9, so there are 10 of 
them, hence you should have 10 neurons in your final layer.
"""

fmnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(64, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
                                    # Try experimenting with this layer
                                    ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy')

model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])

"""
Exercise 5:
Consider the effects of additional layers in the network. What will happen if you add another layer between the one 
with 512 and the final layer with 10.

Click for Answer
Answer
There isn't a significant impact -- because this is relatively simple data. For far more complex data (including color 
images to be classified as flowers that you'll see in the next lesson), extra layers are often necessary.
"""

fmnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(256, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(256, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
                                    ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy')

model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])

"""
Exercise 6:
E6Q1: Consider the impact of training for more or less epochs. Why do you think that would be the case?
Try 15 epochs -- you'll probably get a model with a much better loss than the one with 5
Try 30 epochs -- you might see the loss value decrease more slowly, and sometimes increases. You'll also likely see 
that the results of model.evaluate() didn't improve much. It can even be slightly worse.
This is a side effect of something called 'overfitting' which you can learn about later and it's something you need to 
keep an eye out for when training neural networks. There's no point in wasting your time training if you aren't 
improving your loss, right! :)
"""

fmnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy')

model.fit(training_images, training_labels, epochs=15)  # Experiment with the number of epochs

model.evaluate(test_images, test_labels)

"""
Exercise 7:
Before you trained, you normalized the data, going from values that were 0-255 to values that were 0-1. What would be 
the impact of removing that? Here's the complete code to give it a try. Why do you think you get different results?
"""

fmnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

training_images = training_images / 255.0  # Experiment with removing this line
test_images = test_images / 255.0  # Experiment with removing this line
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)
classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])

"""
Exercise 8:
Earlier when you trained for extra epochs you had an issue where your loss might change. It might have taken a bit of 
time for you to wait for the training to do that, and you might have thought 'wouldn't it be nice if I could stop the 
training when I reach a desired value?' -- i.e. 60% accuracy might be enough for you, and if you reach that after 3 
epochs, why sit around waiting for it to finish a lot more epochs....So how would you fix that? Like any other 
program...you have callbacks! Let's see them in action...
"""


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') >= 0.6:  # Experiment with changing this value
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = MyCallback()

fmnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])
