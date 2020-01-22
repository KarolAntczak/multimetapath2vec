import numpy as np
from tensorflow import keras
from tensorflow_core.python.keras.layers.core import Dense, Flatten
from tensorflow_core.python.keras.layers.embeddings import Embedding
from tensorflow_core.python.keras.models import Sequential


def create_model(embeddings_file: str, input_size, input_length, hidden_size):
    """
    Create simple regression model with a single embedding layer.

    :param embeddings_file: embeddings file to load
    :param input_size: size of input layer
    :param hidden_size: size of embeddings
    :return: Keras model
    """

    model = Sequential()
    model.add(Embedding(input_size, hidden_size, input_length=input_length, name='embedding'))
    model.add(keras.layers.Lambda(lambda x: keras.backend.sum(x, axis=1)))
    #model.add(Flatten())
    model.add(Dense(92, activation="sigmoid"))

    if embeddings_file is not None:
        embeddings = np.loadtxt(embeddings_file)
        model.get_layer("embedding").set_weights([embeddings])

    #model.summary()
    return model

