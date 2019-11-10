import pickle
import numpy as np
from tensorflow_core.python.keras.engine.input_layer import Input
from tensorflow_core.python.keras.layers.core import Reshape, Dense
from tensorflow_core.python.keras.layers.embeddings import Embedding
from tensorflow_core.python.keras.layers.merge import dot

from tensorflow_core.python.keras.models import Model

pairs, labels = pickle.load(open("data/dane.csv.pickle", 'rb'))

pairs = np.asarray(pairs)
labels = np.asarray(labels)
targets = pairs[:,0]
contexts = pairs[:,1]

vocab_size = np.amax(pairs) + 1
vector_dim = 100

input_target = Input((1,))
input_context = Input((1,))
embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')
target = embedding(input_target)
target = Reshape((vector_dim, 1))(target)
context = embedding(input_context)
context = Reshape((vector_dim, 1))(context)
# now perform the dot product operation to get a similarity measure
dot_product = dot([target, context], axes=1)
dot_product = Reshape((1,))(dot_product)
# add the sigmoid output layer
output = Dense(1, activation='sigmoid')(dot_product)

model = Model(inputs=[input_target, input_context], outputs=output)
model.compile(loss='binary_crossentropy', optimizer='rmsprop')
model.summary()
history = model.fit(x=[targets, contexts], y=labels, batch_size=1, epochs=1, validation_split=0.0)

print("Saving the embeddings layer")

weights = model.get_layer("embedding").get_weights()[0]

np.savetxt("models\\embeddings_%d_to_%d.csv" % (vocab_size, vector_dim), weights)

with open('models\\embedding_history', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
