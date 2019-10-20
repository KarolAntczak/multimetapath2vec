import numpy as np
from sklearn.preprocessing import LabelEncoder


def encode_nodes(all_labels, data):
    all_labels = ['empty'] + list(all_labels) # required for Keras skipgram function
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    data_encoded = []
    for row in data:
        walk_encoded = label_encoder.transform(row)
        data_encoded.append(walk_encoded)

    data_encoded = np.asarray(data_encoded)
    return data_encoded
