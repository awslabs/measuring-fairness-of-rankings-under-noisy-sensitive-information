import numpy
import numpy as np
import pandas
import pandas as pd

from data_read.read_embeddings import glove_embeddings


class StringProcessor:
    def __init__(self):
        self.embeddings = None
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'

    def read_embeddings(self):
        self.embeddings = glove_embeddings()

    def process_chars(self, x: pd.DataFrame) -> np.ndarray:
        """
        Return the vectorized string
        :param x: input string
        :return: Numpy array that encodes the input characters
        """

        N = x.shape[0]
        full_names = x['input_string'].copy()

        processed_x = np.array([self.string_vectorizer(full_names.iloc[ell]) for ell in range(N)])
        return processed_x

    def string_vectorizer(self, string: str) -> list:
        """
        Returns the list of character embeddings
        :param string: Input string
        :return: A list containing the embeddings of the input characters
        """
        vector = [[0 if char != letter.lower() else 1 for char in self.alphabet] for letter in string]
        return vector

    def process_words(self, x: pd.DataFrame) -> np.ndarray:
        """
        Returns the vectorized string
        :param x: input string
        :return: numpy array that encodes the input
        """

        # Prepare embedding matrix
        processed_x = np.array([self.vectorize_sentence(x['input_string'].iloc[ell])
                                for ell in range(x.shape[0])])
        return processed_x

    def vectorize_sentence(self, s: str) -> np.ndarray:
        """
        Returns the embeddings of the words in s
        :param s: Input sentence
        :return: Returns the embeddings of the words in s
        """
        words = s.split(' ')
        vector = []
        for word in words:
            embedding_vector = self.embeddings.get(word.lower())
            if embedding_vector is not None:
                vector.append(np.array(embedding_vector))
            else:
                vector.append(np.zeros(50))
        return np.array(vector)

    def avg_embeddings(self, x: pd.DataFrame) -> np.ndarray:
        """
        Computes the average embeddings of the words for each input string
        :param x: Input dataframe
        :return: Embeddings of the input strings
        """
        processed_x = np.array([self.compute_avg_embedding(x['input_string'].iloc[ell]) for ell in range(x.shape[0])])
        return processed_x

    def compute_avg_embedding(self, s: str) -> np.ndarray:
        """
        Computes the average word embedding of the input sentence
        :param s: Input sentence
        :return: Embedding of the input sentence
        """
        words = s.split(' ')
        vector = np.zeros(50)
        count_not_none = 0
        for word in words:
            embedding_vector = self.embeddings.get(word.lower())
            if embedding_vector is not None:
                vector = vector + embedding_vector
                count_not_none += 1
        if count_not_none == 0:
            return vector
        else:
            return vector / count_not_none