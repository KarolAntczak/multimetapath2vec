from functools import partial

import numpy
from sklearn.metrics import f1_score, log_loss, confusion_matrix, accuracy_score
from sklearn.model_selection import KFold

from diagnosis.dataset import load_data
from diagnosis.model import create_model

n_split = 10
epochs = 10
vocab_size = 2238
vector_dim = 100
max_len = 1327

data_file = "../data/dane.csv_2_diagnosis.pickle"

all_x_data, all_y_data = load_data(filename=data_file)

for algorithm in [ "onehot", "node2vec", "metapath2vec", "multimetapath2vec"]:
    print("\tAlgorithm: %s" % algorithm)
    if algorithm == "onehot":
        embeddings_file = None
    else:
        embeddings_file = "..\\models\\embeddings_%s_%d_to_%d.csv" % (algorithm, vocab_size, vector_dim)

    load_model = partial(create_model, input_size=vocab_size,  hidden_size=vector_dim, input_length=max_len, embeddings_file=embeddings_file)
    total_results = []

    for missing_percentage in all_x_data.keys():
        print("%d%% missing data" % missing_percentage)
        x_data = all_x_data[missing_percentage]
        y_data = all_y_data[missing_percentage].reshape((-1, 1))

        fold_size = x_data.size / n_split

        results = []
        k_fold = KFold(n_splits=n_split, shuffle=True, random_state=42)
        fold_no = 0
        for train_index, test_index in k_fold.split(x_data):
            fold_no += 1
            print("\t\tFold %d/%d" % (fold_no, n_split))
            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]

            model = load_model()
            model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=["accuracy"])

            model.fit(x=x_train, y=y_train,  batch_size=1, epochs=10, verbose=0)

            y_pred = model.predict(x_test)
            y_pred = y_pred.round()

            f1_micro = f1_score(y_test, y_pred, average='micro')  # accuracy
            f1_macro = f1_score(y_test, y_pred, average='macro')


            y_pred_class = y_pred > 0.5
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_class).ravel()
            false_negative_rate = fn / (tp + fn)

            accuracy = accuracy_score(y_test, y_pred_class)

            results.append((f1_micro, f1_macro, tn, fp, fn, tp, accuracy))

        avg_results = numpy.average(results, axis=0)
        total_results.append(avg_results)

    print(total_results)
    numpy.savetxt("results/results_%s.csv" % algorithm, total_results, delimiter=';', )
