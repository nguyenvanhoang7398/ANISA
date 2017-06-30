import pandas as pd
import numpy as np
from nltk import sent_tokenize, word_tokenize
from gensim.models.doc2vec import TaggedDocument
import math, pickle, random
from collections import Counter, deque
from nltk import word_tokenize, WordNetLemmatizer
from datetime import datetime
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words

tokenizer = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')
p_stemmer = PorterStemmer()


def time_format(second):
    h = second // 3600
    m = ((second % 3600) // 60)
    s = second % 60
    return [h, m, s]


def save_data(path, py_object):
    with open(path, 'wb') as f:
        pickle.dump(py_object, f)


# Load data from existing pickle
def load_data(pickle_in):
    with open(pickle_in, 'rb') as f:
        contents = pickle.load(f)
    return contents


# args: raw text
# lower, tokenize, eliminate stop words and stem
# return: list of processed tokens
def process_text(raw):
    raw = raw.lower()
    tokens = tokenizer.tokenize(raw)
    stopped_tokens = [i for i in tokens if not i in en_stop and len(i) > 1]
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    return stemmed_tokens


def create_lexicon(fin, fout, upper_bound, lower_bound, word_counts_file, vocab_size):
    print("Create lexicon from", fin)
    lexicon = []

    with open(fin, 'r') as f:
        for line in f:
            complaints = ','.join(line.split(",")[2:])
            lexicon += process_text(complaints)

    word_counts = Counter(lexicon).most_common(vocab_size-1)
    print(word_counts)
    word_counts = dict(word_counts)
    reduced_lexicon = []
    for word in word_counts:
        if upper_bound > word_counts[word] > lower_bound:
            reduced_lexicon.append(word)

    with open(fout, 'wb') as f:
        pickle.dump(reduced_lexicon, f)

    with open(word_counts_file, 'wb') as f:
        pickle.dump(word_counts, f)

    return reduced_lexicon


def create_one_hot_dict(class_list, one_hot_dict_pickle):
    n_class = len(class_list)
    class_dict = {}
    reverse_class_dict = {}

    for i, cl in enumerate(class_list):
        one_hot = np.zeros(n_class)
        one_hot[i] = 1
        one_hot = list(one_hot)
        class_dict[cl] = one_hot
        reverse_class_dict[tuple(one_hot)] = cl

    with open(one_hot_dict_pickle, 'wb') as f:
        pickle.dump((class_dict, reverse_class_dict), f)

    return class_dict, reverse_class_dict


def get_doc_counts(lexicon, dataset, doc_counts_pickle, doc_counts):

    for i, word in enumerate(lexicon):
        for data in dataset:
            if data[0][i] > 0:
                if not word in doc_counts.keys():
                    doc_counts[word] = 1
                else:
                    doc_counts[word] += 1

    with open(doc_counts_pickle, 'wb') as f:
        pickle.dump(doc_counts, f)

    return doc_counts


def get_all_diagnoses(fin, diagnosis_pickle):
    all_diagnoses = []

    with open(fin, 'r', encoding='utf8') as f:
        for line in f:
            diagnosis = line.split(",")[1]
            diagnosis = "_".join(diagnosis.split(" "))
            if diagnosis not in all_diagnoses:
                all_diagnoses += [diagnosis]

    with open(diagnosis_pickle, 'wb') as f:
        pickle.dump(all_diagnoses, f)

    return all_diagnoses


def create_bow_dataset(fin, fout, lexicon, n_samples, class_dict=None, type='direction', recorded_samples=1000):
    print("Create bow dataset from", fin)
    dataset = []
    time0 = datetime.now()

    with open(fin, 'r') as f:
        for i, line in enumerate(f):
            if i % recorded_samples == 0  and i != 0:
                elapsed = (datetime.now() - time0)
                second = elapsed.total_seconds()
                time = time_format(second)
                time_per_sample = float(second) / recorded_samples
                time0 = datetime.now()
                print('Sample run:', i, '/', n_samples, '| Time:', time[0], 'hours',
                      time[1], 'minutes', "%.2f" % time[2], 'seconds.')
                remaining_time = time_format(
                    (time_per_sample * (n_samples - i)))
                print('Estimated remaining time:', remaining_time[0], 'hours', remaining_time[1], 'minutes',
                      "%.2f" % remaining_time[2], 'seconds.')

            complaints = ','.join(line.split(",")[2:])
            current_words = process_text(complaints)
            data = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    idx_val = lexicon.index(word.lower())
                    data[idx_val] += 1

            data = list(data)
            if type == 'diagnosis':
                diagnosis = line.split(",")[1]
                diagnosis = "_".join(diagnosis.split(" "))
                label = class_dict[diagnosis]
            else:
                direction = line.split(",")[0]
                if "ED" in direction:
                    label = [1, 0]
                else:
                    label = [0, 1]
            dataset.append([data, label])

    with open(fout, 'wb') as f:
        pickle.dump(dataset, f)

    return dataset


def create_bow_vector(lexicon, doc):
    current_words = process_text(doc)
    data = np.zeros(len(lexicon))
    for word in current_words:
        if word.lower() in lexicon:
            idx_val = lexicon.index(word.lower())
            data[idx_val] += 1

    return data


# normalize dataset using words appear only in the lexicon
def rescale_and_normalize(lexicon, raw_vector, docs_count, doc_num):

    doc_len = 0
    for w in raw_vector:
        doc_len += w

    for i, word in enumerate(lexicon):
        if raw_vector[i] > 0:
            # tf = raw_vector[i] / doc_len
            idf = np.log(float(doc_num) / docs_count[word])
            # raw_vector[i] *= (tf * idf)
            raw_vector[i] *= (idf / doc_len)

    if np.linalg.norm(raw_vector) == 0:
        return raw_vector
    return (raw_vector / np.linalg.norm(raw_vector)).tolist()


def normalize_dataset(fin, dataset, lexicon, doc_counts, doc_num, normalized_pickle):
    for data in dataset:
        data[0] = rescale_and_normalize(lexicon, data[0], doc_counts, doc_num)

    with open(normalized_pickle, 'wb') as f:
        pickle.dump(dataset, f)

    return dataset


def create_train_test_bow_data(dataset, test_size=0.1):
    random.shuffle(dataset)
    dataset = np.array(dataset)
    testing_size = int(test_size*len(dataset))

    train_inputs = list(dataset[:,0][:-testing_size])
    train_labels = list(dataset[:,1][:-testing_size])
    test_inputs = list(dataset[:,0][-testing_size:])
    test_labels = list(dataset[:,1][-testing_size:])

    return train_inputs, train_labels, test_inputs, test_labels


def sigmoid(x, theta=1.0):
    return 1 / (1 + math.exp(-theta*x))