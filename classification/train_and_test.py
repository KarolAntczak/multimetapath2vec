from functools import partial

import numpy
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

from classification.dataset import load_data
from classification.model import create_model

n_split = 10
epochs = 10
vocab_size = 2238
vector_dim = 100

data_file = "../data/dane.csv_classification.pickle"

x_data, y_data = load_data(filename=data_file)

fold_size = x_data.size / n_split

print("Inputs shape %s" % (x_data.shape,))
print("Outputs shape %s" % (y_data.shape,))

for algorithm in ["multimetapath2vec", "metapath2vec", "node2vec", "onehot"]:
    print("Algorithm: %s" % algorithm)
    if algorithm == "onehot":
        embeddings_file = None
    else:
        embeddings_file = "..\\models\\embeddings_%s_%d_to_%d.csv" % (algorithm, vocab_size, vector_dim)

    load_model = partial(create_model, input_size=vocab_size, hidden_size=vector_dim, output_size=y_data.shape[1],
                         embeddings_file=embeddings_file)
    total_results = []
    for missing_percentage in range(90, 100, 1):
        print("\t%d%% missing data" % missing_percentage)
        results = []
        k_fold = KFold(n_splits=n_split, shuffle=True, random_state=42)
        fold_no = 0
        for train_index, test_index in k_fold.split(x_data):
            fold_no += 1
            print("\t\tFold %d/%d" % (fold_no, n_split))
            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]

            kept_fraction = 1 - (missing_percentage / 100)
            train_size = int(x_train.size * kept_fraction)

            x_train = x_train[0:train_size]
            y_train = y_train[0:train_size]

            model = load_model()
            model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=["accuracy"])

            model.fit(x=x_train, y=y_train,  batch_size=1, epochs=epochs, verbose=0)

            y_pred = model.predict(x_test)
            y_pred_bool = numpy.argmax(y_pred, axis=1)
            y_test_bool = numpy.argmax(y_test, axis=1)

            f1_micro = f1_score(y_test_bool, y_pred_bool, average='micro')  # accuracy
            f1_macro = f1_score(y_test_bool, y_pred_bool, average='macro')

            results.append((f1_micro, f1_macro))

        avg_results = numpy.average(results, axis=0)
        total_results.append(avg_results)

    print(total_results)
    numpy.savetxt("results/results_%s.csv" % algorithm, total_results, delimiter=';', )
