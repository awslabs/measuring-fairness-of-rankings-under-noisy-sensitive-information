import numpy as np

from pathlib import Path


def glove_embeddings():
    """
    Reads GloVe embeddings (with 50 dimensions)
    Reference: https://nlp.stanford.edu/projects/glove/
    :return: GloVe embeddings
    """
    glove_file = Path(__file__).parent.parent.parent.parent.joinpath("bin").joinpath("data").\
        joinpath("glove.6B.50d.txt")
    embeddings_index = {}
    with open(glove_file) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    print("Found %s word vectors." % len(embeddings_index))
    return embeddings_index
