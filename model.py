# this model refers to the model class we developed which is model.py
# from model import Model

# imports
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout, GRU, Bidirectional
import numpy as np
from keras.models import load_model


class Model:
    def __init__(self) -> None:
        self.optimizer = tf.keras.optimizers.legacy.Adam
        self.metrices = ["accuracy"]
        self.elipson = None
        self.__model = None
        self.model_name = None

    def init_model(
        self,
        seq_len,
        vocab_size=None,
        model_type: "LSTM" = "LSTM or GRU or Bidirectional",
        units=64,
        no_of_neurons=8,
        learning_rate=0.01,
        layer=None,
        droup_out=0.2,
        model_name=None,
    ):
        """initialize the neural network model with the given characterstices like model type, units , sequncence-length, vocab-size and learning rate"""

        # assigning the name
        self.model_name = model_name  # for saving the model

        # checking and loading the previous model
        try:
            self.__model = self.load_model(self.model_name)
        except OSError:
            print("previous model not found creating new model")

            # creating the new model
            self.__model = Sequential()
            self.__model.add(
                Embedding(input_dim=vocab_size, output_dim=30, input_length=seq_len)
            )
            if layer is not None:
                self.__model.add(model_type(layer(units=units)))
            else:
                self.__model.add(model_type(units=units))

            self.__model.add(Dropout(droup_out))
            # self.__model.add(LSTM(units=(units//2)))
            # self.__model.add(Dense(self.neurons, activation=self.activation_func))
            self.__model.add(Dense(2, activation="softmax"))
            self.__model.compile(
                loss="categorical_crossentropy",
                optimizer=self.optimizer(learning_rate),
                metrics=self.metrices,
            )
        else:
            # checking and loading the model wiights
            try:
                self.load_model_weights(self.model_name)
            except tf.errors.NotFoundError:
                print("Model weight not found")
        finally:
            # finally compiling the model to match the current hyper parameters
            self.__model.compile(
                loss="categorical_crossentropy",
                optimizer=self.optimizer(learning_rate),
                metrics=self.metrices,
            )

        print(self.__model.summary())

    def init_general_model(
        self,
        hidden_layers,
        no_of_neuraons,
        activation_func="relu",
        learning_rate=0.01,
        model_name="general",
    ):
        """initialize the neural network model with the given characterstices like layer"""
        # initialize the model name
        self.model_name = model_name

        # checking the previous model if exists and also checking for it's weights if not creating new model
        try:
            self.__model = self.load_model(self.model_name)
        except tf.errors.NotFoundError:
            print("previous model not found creating new model")

            # creating the new general model
            self.__model = Sequential()
            self.__model.add(
                Dense(no_of_neuraons // 2, input_dim=70, activation=activation_func)
            )

            for _ in range(hidden_layers):
                self.__model.add(Dense(no_of_neuraons, activation=activation_func))
            # self.__model.add(Dropout(0.2))
            self.__model.add(Dense(2, activation="softmax"))

        else:
            # checking and loading the model wiights
            try:
                self.load_model_weights(self.model_name)
            except tf.errors.NotFoundError:
                print("Model weight not found")
        finally:
            # compiling the model in any case
            self.__model.compile(
                loss="categorical_crossentropy",
                optimizer=self.optimizer(learning_rate),
                metrics=self.metrices,
            )

        print(self.__model.summary())

    def train_model(self, X_train, y_train, epochs=100, batch_size=500, verbose=2):
        """train the model using the traning set and default epoch is set to 100, batch_size is 500 and verbose is 2 which can be overridden"""
        self.__model.fit(
            X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose
        )

        # saving the model weights
        self.save_model_weights()

        # save the model
        self.save_model()

        return self.__model

    def evaluate_model(self, X_evaluate, y_evaluate, verbose=2):
        """evaluate the model after the traning have to done to the modfel"""
        self.__model.evaluate(X_evaluate, y_evaluate, verbose=verbose)
        return self.__model

    def test_model(self, X_test):
        """returns the prediction from the model trained"""
        return self.__model.predict(X_test, verbose=2)

    def predict_review(self, review):
        """predict the review is whether postive or negative after traning the model"""
        return self.__model.predict(review, verbose=2)

    def save_model(self, path="./data/model_weights/"):
        """this will save the entire model to the file"""
        self.__model.save((path + self.model_name))

    def load_model(self, model_name, path="./data/model_weights/"):
        """loads the model that is being saved"""
        return load_model(f"{path}{model_name}")

    def save_model_weights(self, path="./data/model_weights/"):
        """save the model into the file when called"""
        self.__model.save_weights(f"{path}{self.model_name}_weights")

    def load_model_weights(self, model_name, path="./data/model_weights/"):
        """load the model from the the file when called"""
        self.__model.load_weights(f"{path}{model_name}_weights")
