import warnings
from functools import partial

import numpy
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score

from data_encoding import to_binary
from diagnosis.dataset import load_data
from diagnosis.model import create_model
warnings.filterwarnings('ignore')

n_split = 10
epochs = 10
vocab_size = 2238
vector_dim = 100
max_len = 10

data_file = "../data/dane.csv_diagnosis.pickle"

x_train_all, y_train_all, x_val_all, y_val_all = load_data(filename=data_file)

for algorithm in ["metapath2vec", "node2vec", "multimetapath2vec", "onehot", ]:
    print("Algorithm: %s" % algorithm)
    if algorithm == "onehot":
        embeddings_file = None
    else:
        embeddings_file = "..\\models\\embeddings_%s_%d_to_%d.csv" % (algorithm, vocab_size, vector_dim)

    load_model = partial(create_model, input_size=vocab_size,  hidden_size=vector_dim, input_length=max_len,
                         embeddings_file=embeddings_file)
    total_results = []

    for missing_percentage in range(90, 100, 1):
        print("\t%d%% missing data" % missing_percentage)
        results = []
        x_train = pad_sequences(x_train_all[missing_percentage], maxlen=10)
        x_val = pad_sequences(x_val_all[missing_percentage], maxlen=10)
        y_train = to_binary(y_train_all[missing_percentage], max_value=91)
        y_val = to_binary(y_val_all[missing_percentage], max_value=91)

        x_train_folds = numpy.array_split(x_train, 10)
        x_val_folds = numpy.array_split(x_val, 10)
        y_train_folds = numpy.array_split(y_train, 10)
        y_val_folds = numpy.array_split(y_val, 10)

        for fold in range(0, 10):
            x_train_fold = x_train_folds[fold]
            y_train_fold = y_train_folds[fold]
            x_val_fold = x_val_folds[fold]
            y_val_fold = y_val_folds[fold]
            model = load_model()
            model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])

            #model.fit(x=x_train_fold, y=y_train_fold,  validation_data=(x_val_fold, y_val_fold), batch_size=4, epochs=10, verbose=0)
            model.fit(x=x_train_fold, y=y_train_fold,  batch_size=4, epochs=10, verbose=0)

            y_pred = model.predict(x_val_fold)

            y_pred_bool = numpy.argmax(y_pred, axis=1)
            y_test_bool = numpy.argmax(y_val_fold, axis=1)

            f1_micro = f1_score(y_test_bool, y_pred_bool, average='micro')  # accuracy
            f1_macro = f1_score(y_test_bool, y_pred_bool, average='macro')

            results.append((f1_micro, f1_macro))
            print("\t\tFold %d/10:\t%.3f\t%.3f" % (fold+1, f1_micro, f1_macro))

        avg_results = numpy.average(results, axis=0)
        total_results.append(avg_results)

    numpy.savetxt("results/results_%s.csv" % algorithm, total_results, delimiter=';', )
