from bag_of_word import BOW
from utils import create_lexicon, create_bow_dataset, create_train_test_bow_data, load_data, save_data
from sklearn import svm
from samples import new_docs


def train_direction():
    doc_counts = load_data("data/doc_counts.pickle")

    bow_dataset = []
    for i in range(5):
        bow_dataset += load_data("data/direction_normalized_dataset_" + str(i + 1) + ".pickle")

    train_inputs, train_labels, test_inputs, test_labels = create_train_test_bow_data(dataset=bow_dataset,
                                                                                      test_size=0.1)
    bow = BOW(n_nodes_hl1=500, n_nodes_hl2=500, n_classes=2, n_docs=20545,
              lexicon_pickle="data/direction_lexicon.pickle",
              dataset=bow_dataset, doc_counts=doc_counts, path="data/model_direction/",
              model_name="model_direction", log="data/model_direction/model_direction_log.pickle")
    bow.train_nn_bow(train_inputs=train_inputs, train_labels=train_labels, batch_size=128, n_recorded_batch=500,
                     n_epoch=15)
    bow.test_nn_bow(test_inputs=test_inputs, test_labels=test_labels, category="binary")
    save_data("data/model_direction/unseen_test_sets.pickle", (test_inputs, test_labels))


def test_direction(threshold=None):
    doc_counts = load_data("data/doc_counts.pickle")

    bow = BOW(n_nodes_hl1=500, n_nodes_hl2=500, n_classes=2, n_docs=20545,
              lexicon_pickle="data/lexicon.pickle",
              dataset=[], doc_counts=doc_counts, path="data/model_direction/",
              model_name="model_direction", log="data/model_direction/model_direction_log.pickle")
    test_inputs, test_labels = load_data("data/model_direcion/unseen_test_sets.pickle")
    bow.test_nn_bow(test_inputs=test_inputs, test_labels=test_labels, category="binary", threshold=threshold)


def train_diagnosis():
    doc_counts = load_data("data/doc_counts.pickle")

    bow_dataset = []
    for i in range(5):
        bow_dataset += load_data("data/diagnosis_normalized_dataset_" + str(i+1) + ".pickle")

    train_inputs, train_labels, test_inputs, test_labels = create_train_test_bow_data(dataset=bow_dataset,
                                                                                 test_size=0.1)

    bow = BOW(n_nodes_hl1=500, n_nodes_hl2=500, n_classes=19, n_docs=20545,
              lexicon_pickle="data/lexicon.pickle",
              dataset=bow_dataset, doc_counts=doc_counts ,path="data/model_diagnosis/",
              model_name="model_diagnosis", log="data/model_diagnosis/model_diagnosis_log.pickle")

    bow.train_nn_bow(train_inputs=train_inputs, train_labels=train_labels, batch_size=128, n_recorded_batch=500,
                     n_epoch=15)
    bow.test_nn_bow(test_inputs=test_inputs, test_labels=test_labels)
    save_data("data/model_diagnosis/unseen_test_sets.pickle", (test_inputs, test_labels))


def test_diagnosis():
    doc_counts = load_data("data/doc_counts.pickle")

    bow = BOW(n_nodes_hl1=500, n_nodes_hl2=500, n_classes=19, n_docs=20545,
              lexicon_pickle="data/lexicon.pickle",
              dataset=[], doc_counts=doc_counts, path="data/model_diagnosis/",
              model_name="model_diagnosis", log="data/model_diagnosis/model_diagnosis_log.pickle")
    test_inputs, test_labels = load_data("data/model_diagnosis/unseen_test_sets.pickle")
    bow.test_nn_bow(test_inputs=test_inputs, test_labels=test_labels)


# diagnosis
def diagnose():
    doc_counts = load_data("data/doc_counts.pickle")
    class_dict, reverse_class_dict = load_data("data/one_hot_dict.pickle")

    bow = BOW(n_nodes_hl1=500, n_nodes_hl2=500, n_classes=19, n_docs=20545, lexicon_pickle="data/lexicon.pickle",
              dataset=[], doc_counts=doc_counts ,path="data/model_diagnosis/",
              model_name="model_diagnosis", log="data/model_diagnosis/model_diagnosis_log.pickle")
    diagnoses = bow.predict(new_docs, reverse_class_dict, top_n=5)
    for i, diag in enumerate(diagnoses):
        print("...Text:", new_docs[i])
        print(diag)


def direct(threshold=None):
    doc_counts = load_data("data/doc_counts.pickle")

    bow_direct = BOW(n_nodes_hl1=500, n_nodes_hl2=500, n_classes=2, n_docs=20545, lexicon_pickle="data/lexicon.pickle",
                         dataset=[], doc_counts=doc_counts, path="data/model_direction/",
                         model_name="model_direction", log="data/model/model_direction_log.pickle")
    reverse_direct_dict = {
            (1, 0): "ED",
            (0, 1): "GP"
        }
    direction = bow_direct.predict(new_docs, reverse_direct_dict, top_n=1, category="binary", threshold=threshold)
    for i, dir in enumerate(direction):
        print("...Text:", new_docs[i])
        print(dir)