import numpy as np
import tensorflow as tf 

import tensorflow_datasets as tfds

mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)

mnist_train , mnist_test = mnist_dataset['train'], mnist_dataset['test']

#Take 10% of the training data to create a validation dataset. 
#num_examples specifies the number of samples (images) in the training data
num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples, tf.int64)

#Store the number of test samples in a dedicated variable instead of using mnist_info 
num_test_samples = mnist_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples, tf.int64)

#Scale the data so that the results are more numerically stable. We want the results to be between 0 and 1. 

def scale(image,label):

  image = tf.cast(image, tf.float32)
  image /= 255.

  return image,label 

scaled_train_and_validation_data = mnist_train.map(scale)
test_data = mnist_test.map(scale)

#Actually take the 10% of the test train data for the validation data. We do this with the .take() method
#.take() method takes the first 6000 from shuffled_train_and_validation_data 
#.skip() method skips the first 6000 from shuffled_train_and_validation_data and takes the rest 
validation_data = shuffled_train_and_validation_data.take(num_validation_samples)
train_data = shuffled_train_and_validation_data.skip(num_validation_samples)

BATCH_SIZE = 150

train_data = train_data.batch(BATCH_SIZE)

validation_data = validation_data.batch(num_validation_samples)

test_data = test_data.batch(num_test_samples)

validation_inputs, validation_targets = next(iter(validation_data))

input_size = 784
output_size = 10
hidden_layer_size = 2000

model = tf.keras.Sequential([
                             tf.keras.layers.Flatten(input_shape=(28,28,1)),
                             #tf.keras.layers.Dense is basically implementing: output = activation(dot(input,weight)+bias)
                             tf.keras.layers.Dense(hidden_layer_size, activation='relu'), #1st hidden layer
                             tf.keras.layers.Dense(hidden_layer_size, activation='relu'), #2nd hidden layer
                             tf.keras.layers.Dense(hidden_layer_size, activation='relu'), #3rd hidden layer
                             tf.keras.layers.Dense(hidden_layer_size, activation='relu'), #4th hidden layer
                             tf.keras.layers.Dense(hidden_layer_size, activation='relu'), #5th hidden layer
                             tf.keras.layers.Dense(output_size, activation='softmax') #output layer

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

NUM_EPOCHS = 10

model.fit(train_data, epochs=NUM_EPOCHS, validation_data=(validation_inputs, validation_targets), verbose=2)

test_loss, test_accuracy = model.evaluate(test_data)

print('Test loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))
