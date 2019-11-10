import pickle

import numpy as np
from tensorflow_core.python.keras.layers.core import Dense, Flatten
from tensorflow_core.python.keras.layers.embeddings import Embedding
from tensorflow_core.python.keras.models import Sequential

from data_encoding import to_binary

x_train, y_train, x_test, y_test = pickle.load(open("../data/dane.csv_classification.pickle", 'rb'))
y = np.concatenate((y_train, y_test))

y_train = to_binary(y_train, np.amax(y))
y_test = to_binary(y_test, np.amax(y))


vocab_size = 2238
vector_dim = 100
embeddings = np.loadtxt("..\\models\\embeddings_%d_to_%d.csv" % (vocab_size, vector_dim))

print("Inputs shape %s" % (x_train.shape,))
print("Outputs shape %s" % (y_train.shape,))
print("Embeddings shape %s" % (embeddings.shape,))


# Logistic regression model
model = Sequential()

model.add(Embedding(vocab_size, vector_dim, input_length=1, name='embedding'))
model.get_layer("embedding").set_weights([embeddings])
model.add(Flatten())
model.add(Dense(y_train.shape[1], activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics = ["accuracy"])
model.summary()
print( model.evaluate(x_test, y_test))
model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=1, epochs=100, verbose=2)
