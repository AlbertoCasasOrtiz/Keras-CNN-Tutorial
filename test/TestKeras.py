# This code has been taken from the following source and modified, organized in class functions, for more clarity.
#
# Source: https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5

from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten


class TestKeras:

    def __init__(self):
        # download mnist data and split into train and test sets
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        # create model
        self.model = Sequential()

    def explore_dataset(self):
        # plot the first image in the dataset
        plt.imshow(self.X_train[0])
        # check image shape
        print(self.X_train[0].shape)

    def preproccess_data(self):
        # reshape data to fit model
        self.X_train = self.X_train.reshape(60000, 28, 28, 1)
        self.X_test = self.X_test.reshape(10000, 28, 28, 1)
        # one-hot encode target column
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)
        print(self.y_train[0])

    def build_model(self):
        # add model layers
        self.model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
        self.model.add(Conv2D(32, kernel_size=3, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(10, activation='softmax'))

    def compile_model(self):
        # compile model using accuracy to measure model performance
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self):
        # train the model
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=3)

    def make_predictions(self):
        # predict first 4 images in the test set
        print(self.model.predict(self.X_test[:4]))

        # actual results for first 4 images in test set
        print(self.y_test[:4])
