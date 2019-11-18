import numpy as np
from tensorflow_core.python.keras.layers.core import Dense, Flatten
from tensorflow_core.python.keras.layers.embeddings import Embedding
from tensorflow_core.python.keras.models import Sequential


def load_model(embeddings_file,  output_size):
    # Logistic regression model

    model = Sequential()
    embeddings = np.loadtxt(embeddings_file)
    model.add(Embedding(embeddings.shape[0], embeddings.shape[1], input_length=1, name='embedding'))
    model.get_layer("embedding").set_weights([embeddings])
    model.add(Flatten())
    model.add(Dense(output_size, activation="sigmoid"))
    return model
