import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from data_read.read_embeddings import glove_embeddings
from data_read.real_dataset_builder import RealFIFADatasetBuilder
from label_predicting.string_preprocessing import StringProcessor


class LSTMModel:
    def __init__(self):
        self.model = None
        self.model_python = None
        self.str_process_obj = None

    def train(self, x: pd.DataFrame, y: np.array, word_level=False) -> "LSTMModel":
        """
        Trains an LSTM model

        :param x: Train data
        :param y: Target values
        :return: Returns itself (a trained model)
        """

        # padding rows in x to ensure equal length
        N = x.shape[0]
        batch_size = 64
        epochs = 30

        # Processing strings
        self.str_process_obj = StringProcessor()

        # Vectorizing the strings, either character-level or word-level
        if word_level:
            self.str_process_obj.read_embeddings()
            processed_x = self.str_process_obj.process_words(x)
            print(f"in lstm processed_data shape {processed_x.shape}")
        else:
            processed_x = self.str_process_obj.process_chars(x)

        self.model = Sequential()
        # 16 for goodreads, 64 Fifa
        if not word_level:
            self.model.add(LSTM(64, input_shape=(processed_x.shape[1], processed_x.shape[2]), dropout=0.25,
                                recurrent_dropout=0.2))
        else:
            batch_size = 1024
            self.model.add(LSTM(64, input_shape=(processed_x.shape[1], processed_x.shape[2]), dropout=0.1,
                                recurrent_dropout=0.1))
        self.model.add(Dense(1, activation='sigmoid'))
        if word_level:
            opt = Adam(learning_rate=1)
            self.model.compile(loss='binary_crossentropy', optimizer=opt)
        else:
            self.model.compile(loss='binary_crossentropy', optimizer="adam")

        self.model.fit(processed_x, y, batch_size=batch_size, epochs=epochs, validation_split=0, verbose=0)

        return self

    def predict(self, x: pd.DataFrame, y: pd.Series, word_level=False) -> pd.Series:
        """
        Predicts labels for the given data

        :param x: The data whose attribute is predicted with the given model
        :param y: Ground truth labels
        :param word_level: word-level process if True, otherwise char-level
        :return: Predicted labels
        """

        if word_level:
            processed_x = self.str_process_obj.process_words(x)
        else:
            processed_x = self.str_process_obj.process_chars(x)

        # print(x)
        output_y = self.model.predict(processed_x)
        output_y = np.where(output_y > 0.5, 1, 0)
        output_y = output_y.reshape((output_y.shape[0],))
        return pd.Series(output_y)


