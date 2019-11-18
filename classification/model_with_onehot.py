from tensorflow_core.python.keras.layers.core import Dense, Flatten
from tensorflow_core.python.keras.layers.embeddings import Embedding
from tensorflow_core.python.keras.models import Sequential


def load_model(input_size, hidden_size, output_size):
    # Logistic regression model
    model = Sequential()
    model.add(Embedding(input_size, hidden_size, input_length=1, name='embedding'))
    model.add(Flatten())
    model.add(Dense(output_size, activation="sigmoid"))

    return model
