# imports
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.utils.data_utils import pad_sequences
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import load_model


class Model:
    def __init__(self, no_of_neurons=8, no_of_hidden_layers=4, activation_function='relu',) -> None:
        self.learning_rate = 0.01
        self.hidden_layers = no_of_hidden_layers
        self.neurons = no_of_neurons
        self.optimizer = tf.keras.optimizers.legacy.Adam(self.learning_rate)
        self.metrices = ['accuracy']
        self.activation_func = activation_function
        self.elipson = None
        self.__model = None

    def recurrent_model(self, vocab_size, seq_len, units=64):
        """initialize the neural network model with the given characterstices like layer"""
        self.__model = Sequential()
        self.__model.add(Embedding(vocab_size, seq_len, input_length=seq_len))
        self.__model.add(LSTM(units=units))
        self.__model.add(Dropout(0.5))
        # self.__model.add(LSTM(units=(units//2)))
        self.__model.add(Dense(self.neurons, activation=self.activation_func))
        self.__model.add(Dense(2, activation='softmax'))
        self.__model.compile(loss='categorical_crossentropy',
                             optimizer='adam', metrics=['accuracy'])

        print(self.__model.summary())

    def general_model(self):
        """initialize the neural network model with the given characterstices like layer"""
        self.__model = Sequential()
        self.__model.add(Dense(self.neurons, input_dim=138,
                         activation=self.activation_func))
        for _ in range(self.hidden_layers):
            self.__model.add(
                Dense(self.neurons, activation=self.activation_func))
        self.__model.add(Dropout(0.2))
        self.__model.add(Dense(2, activation='softmax'))
        self.__model.compile(loss='categorical_crossentropy',
                             optimizer=self.optimizer, metrics=self.metrices)

        # load the previous trained weights of the model
        self.load_model_weights()

    def train_model(self, X_train, y_train, epochs=100, batch_size=500, verbose=2):
        """train the model using the traning set and default epoch is set to 100, batch_size is 500 and verbose is 2 which can be overridden"""
        self.__model.fit(X_train, y_train, epochs=epochs,
                         batch_size=batch_size, verbose=verbose)

        # saving the model weights
        self.save_model_weights()

        # save the model
        self.save_model()

        return self.__model

    def evaluate_model(self, X_evaluate, y_evaluate, verbose=2):
        """evaluate the model after the traning have to done to the modfel """
        self.__model.evaluate(X_evaluate, y_evaluate, verbose=verbose)
        return self.__model

    def test_model(self, X_test):
        """returns the prediction from the model trained"""
        return self.__model.predict(X_test, verbose=2)

    def predict_review(self, review):
        """predict the review is whether postive or negative after traning the model"""
        return self.__model.predict(review, verbose=2)

    def save_model(self):
        """this will save the entire model to the file"""
        self.__model.save("./data/model_weights/model.h5")

    def load_model(self, path="./data/model_weights/weights.h5"):
        """loads the model that is being saved"""
        return load_model(path)

    def save_model_weights(self):
        """save the model into the file when called"""
        self.__model.save_weights("./data/model_weights/weights.h5")

    def load_model_weights(self):
        """load the model from the the file when called"""
        self.__model.load_weights('./data/model_weights/weights.h5')

    def tokenize_and_pad_items(self, item_seq=None, max_length=70):
        """tokenize the items, converts to the sequence and returns the sequence"""
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(item_seq)
        item_seq = tokenizer.texts_to_sequences(item_seq)
        item_seq = pad_sequences(item_seq, maxlen=max_length, truncating='pre')
        return item_seq
