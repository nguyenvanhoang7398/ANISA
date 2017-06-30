from utils import create_lexicon, get_all_diagnoses, create_one_hot_dict, create_bow_dataset, \
    get_doc_counts, normalize_dataset, load_data

lexicon = create_lexicon(fin="data/data_model1.csv", fout="data/lexicon.pickle",
                         word_counts_file="data/word_counts.pickle",
                         upper_bound=50000, lower_bound=20, vocab_size=20000)
print(len(lexicon))
all_diagnoses = get_all_diagnoses(fin="data/data_model1.csv", diagnosis_pickle="data/all_diagnoses.pickle")
class_dict, reverse_class_dict = create_one_hot_dict(class_list=all_diagnoses,
                                                     one_hot_dict_pickle="data/one_hot_dict.pickle")
dataset_direction = []
for i in range(5):
    bow_dataset_diagnosis = create_bow_dataset(fin="data/data_model1_" + str(i+1) + ".csv",
                                               fout="data/diagnosis_dataset_" + str(i+1) + ".pickle",
                                               lexicon=lexicon, n_samples=4109, class_dict=class_dict, type='diagnosis')
    bow_dataset_direction = create_bow_dataset(fin="data/data_model1_" + str(i+1) + ".csv",
                                               fout="data/direction_dataset_" + str(i+1) + ".pickle",
                                               lexicon=lexicon, n_samples=4109, class_dict=None, type='direction')
    dataset_direction += bow_dataset_direction

doc_counts = get_doc_counts(lexicon=lexicon, dataset=dataset_direction,
                            doc_counts_pickle="data/doc_counts.pickle", doc_counts={})

for i in range(5):
    bow_dataset_diag = load_data("data/diagnosis_dataset_" + str(i+1) + ".pickle")
    normalized_dataset_diag = normalize_dataset(fin="data/data_model1_" + str(i+1) + ".csv", dataset=bow_dataset_diag,
                                                lexicon=lexicon, doc_counts=doc_counts, doc_num=4109,
                                                normalized_pickle="data/diagnosis_normalized_dataset_" + str(i+1) + ".pickle")
    bow_dataset_direct = load_data("data/direction_dataset_" + str(i+1) + ".pickle")
    normalized_dataset_direct = normalize_dataset(fin="data/data_model1_" + str(i+1) + ".csv", dataset=bow_dataset_direct,
                                                lexicon=lexicon, doc_counts=doc_counts, doc_num=4109,
                                                normalized_pickle="data/direction_normalized_dataset_" + str(i+1) + ".pickle")
