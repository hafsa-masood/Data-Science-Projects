# MNIST Machine Learning Model

This is my first machine learning project that I worked on during my data science course. I made a machine learning model based on the MNIST dataset. The MNIST dataset consists of 70,000 images of handwritten digits, 0 to 9. The goal of this algorithm is to take an image as an input, and correctly determine which digit is shown on the image as the output.

NOTE: This ML model was made in Google Colab. 

#### Processing the data: 

First, I split the data set into three subsets: training, validation, and testing. Tensorflow datasets don’t have a validation dataset by default, so I used 10% of the training data to use for validation purposes. 

Next, I scaled the data so that the results could be between 0 and 1, to keep the weights in the neural network small, hence making it more numerically stable. After the data was scaled, it was shuffled.

#### Building the model: 

After the data was processed, I built the neural network, specifying the input, output, and hidden layer size. I added as many hidden layers as I saw appropriate. In my case, I chose 5 hidden layers. 

```python
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
```                             

Following that, I selected an optimizer and loss function.

```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

The last step was to finally train the model. I specified the number of epochs I wanted the algorithm to run for, and let the program run. 

```python
NUM_EPOCHS = 10

model.fit(train_data, epochs=NUM_EPOCHS, validation_data=(validation_inputs, validation_targets), verbose=2)
```

These are the results I got:

```
Epoch 1/10
360/360 - 144s - loss: 0.2478 - accuracy: 0.9266 - val_loss: 0.1334 - val_accuracy: 0.9612 - 144s/epoch - 400ms/step
Epoch 2/10
360/360 - 132s - loss: 0.1079 - accuracy: 0.9703 - val_loss: 0.1331 - val_accuracy: 0.9643 - 132s/epoch - 367ms/step
Epoch 3/10
360/360 - 132s - loss: 0.0874 - accuracy: 0.9758 - val_loss: 0.0859 - val_accuracy: 0.9773 - 132s/epoch - 368ms/step
Epoch 4/10
360/360 - 131s - loss: 0.0673 - accuracy: 0.9817 - val_loss: 0.0798 - val_accuracy: 0.9763 - 131s/epoch - 363ms/step
Epoch 5/10
360/360 - 131s - loss: 0.0575 - accuracy: 0.9839 - val_loss: 0.0886 - val_accuracy: 0.9773 - 131s/epoch - 364ms/step
Epoch 6/10
360/360 - 131s - loss: 0.0527 - accuracy: 0.9859 - val_loss: 0.0605 - val_accuracy: 0.9825 - 131s/epoch - 363ms/step
Epoch 7/10
360/360 - 131s - loss: 0.0427 - accuracy: 0.9884 - val_loss: 0.0681 - val_accuracy: 0.9815 - 131s/epoch - 364ms/step
Epoch 8/10
360/360 - 131s - loss: 0.0361 - accuracy: 0.9903 - val_loss: 0.0659 - val_accuracy: 0.9855 - 131s/epoch - 364ms/step
Epoch 9/10
360/360 - 131s - loss: 0.0359 - accuracy: 0.9898 - val_loss: 0.0456 - val_accuracy: 0.9877 - 131s/epoch - 363ms/step
Epoch 10/10
360/360 - 131s - loss: 0.0315 - accuracy: 0.9914 - val_loss: 0.0421 - val_accuracy: 0.9898 - 131s/epoch - 363ms/step
```

#### Testing the model: 

After ensuring that the accuracy of my model was increasing and the loss was decreasing, I ran the algorithm using the test data. 

```python
test_loss, test_accuracy = model.evaluate(test_data)
```

#### Comments about the results:

I have learned that building a machine learning model that is both accurate and efficient comes down to a tug of war between computational speed and accuracy. 

In my first execution of this algorithm, I assumed a neural network with a greater width and depth would provide increased accuracy. I attempted to train my model with 10 hidden layers and a hidden layer width of 5000. I hypothesize that this factor caused my model’s training to time out and stop executing at the 6 hour mark.

For my second run, I decided to sacrifice a small amount of accuracy for efficiency. I halved the number of hidden layers, making it 5, and reduced the hidden layer size to 2000. After the changes I made, the time it took for training my model went from around six hours to 20 minutes. After running the test data through the algorithm, the final loss was 0.11 and the accuracy was 97.72%. This means that the ML model will correctly determine which numerical digit is shown on an image 97.72% of the time. 

I will continue attempting to increase the accuracy of this model. Stay tuned.  

If you are curious, check out the [full code here]()!


