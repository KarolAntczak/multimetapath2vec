import pickle

import numpy as np
from tensorflow_core.python.keras.layers.core import Dense
from tensorflow_core.python.keras.models import Sequential

from data_encoding import to_binary

x_train, y_train, x_test, y_test = pickle.load(open("../data/dane.csv_classification.pickle", 'rb'))
x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))

x_train = to_binary(x_train, np.amax(x))
x_test = to_binary(x_test, np.amax(x))
y_train = to_binary(y_train, np.amax(y))
y_test = to_binary(y_test, np.amax(y))

print("Inputs shape %s" % (x_train.shape,))
print("Outputs shape %s" % (y_train.shape,))

# Logistic regression model
model = Sequential()
model.add(Dense(y_train.shape[1], input_dim=x_train.shape[1], activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics = ["accuracy"])
model.summary()

print( model.evaluate(x_test, y_test))
model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=1, epochs=100, verbose=2)
