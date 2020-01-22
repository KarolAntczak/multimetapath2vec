import numpy as np
from sklearn.preprocessing import LabelEncoder


def to_integers(all_labels, data):
    """
    Encode data into vectors of integers based on labels
    :param all_labels: list of all possible labels
    :param data: list of input vectors
    :return:
    """
    all_labels = ['empty'] + list(all_labels) # required for Keras skipgram function
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    data_encoded = []
    for row in data:
        row_encoded = label_encoder.transform(row)
        data_encoded.append(row_encoded)

        assert len(row_encoded) == len(row)
    data_encoded = np.asarray(data_encoded)

    return data_encoded


def to_binary(data, max_value):
    """
    Convert vectors of numbers into binary vectors,
    with ones at indices specified by these numbers.

    :param data: list of vectors containing numbers
    :return: list of binary vectors
    """
    encoded_data = []

    for row in data:
        encoded_row = np.zeros(max_value+1)
        encoded_row[row] = 1.
        encoded_data.append(encoded_row)
    encoded_data = np.asarray(encoded_data)
    return encoded_data




