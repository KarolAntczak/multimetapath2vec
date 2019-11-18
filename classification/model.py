import numpy as np
from tensorflow_core.python.keras.layers.core import Dense, Flatten
from tensorflow_core.python.keras.layers.embeddings import Embedding
from tensorflow_core.python.keras.models import Sequential


def create_model(embeddings_file: str, input_size, hidden_size, output_size):
    """
    Create simple regression model with a single embedding layer.

    :param embeddings_file: embeddings file to load
    :param input_size: size of input layer
    :param hidden_size: size of embeddings
    :param output_size: size of output layer
    :return: Keras model
    """
    model = Sequential()
    model.add(Embedding(input_size, hidden_size, input_length=1, name='embedding'))
    if embeddings_file is not None:
        embeddings = np.loadtxt(embeddings_file)
        model.get_layer("embedding").set_weights([embeddings])
    model.add(Flatten())
    model.add(Dense(output_size, activation="sigmoid"))
    return model

