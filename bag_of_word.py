import tensorflow as tf
import numpy as np
from utils import load_data, time_format, create_bow_vector, rescale_and_normalize, sigmoid
import pickle
from datetime import datetime
from sklearn import metrics
from collections import Counter


class BOW:
    def __init__(self, n_nodes_hl1, n_nodes_hl2, n_classes, n_docs, lexicon_pickle, dataset,
                 doc_counts, model_name="bow_model", path="bow_model/",
                 log="bow_log.pickle"):
        self.n_node_hl1 = n_nodes_hl1
        self.n_node_hl2 = n_nodes_hl2
        self.n_classes = n_classes
        self.n_docs = n_docs
        self.lexicon = load_data(lexicon_pickle)
        self.lexicon_size = len(self.lexicon)
        self.dataset = dataset
        self.doc_counts = doc_counts
        self.model_name = model_name
        self.path = path
        self.log = log

        self.x = tf.placeholder('float')
        self.y = tf.placeholder('float')

        self.hidden_1_layer = {'f_fum': n_nodes_hl1,
                          'weights': tf.Variable(tf.random_normal([self.lexicon_size, n_nodes_hl1])),
                          'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

        self.hidden_2_layer = {'f_fum': n_nodes_hl2,
                          'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                          'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

        self.output_layer = {'f_fum': None,
                        'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                        'biases': tf.Variable(tf.random_normal([n_classes])), }

        self.saver = tf.train.Saver()

    def create_model(self, data):
        l1 = tf.add(tf.matmul(data, self.hidden_1_layer['weights']), self.hidden_1_layer['biases'])
        l1 = tf.nn.sigmoid(l1)
        l2 = tf.add(tf.matmul(l1, self.hidden_2_layer['weights']), self.hidden_2_layer['biases'])
        l2 = tf.nn.sigmoid(l2)
        output = tf.matmul(l2, self.output_layer['weights']) + self.output_layer['biases']
        return output

    def train_nn_bow(self, train_inputs, train_labels, n_epoch=10, batch_size=128, n_recorded_batch=100):
        prediction = self.create_model(self.x,)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, self.y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            try:
                with open(self.log, 'rb') as f:
                    epoch = pickle.load(f) + 1
                    print('Starting with epoch', epoch)

            except:
                epoch = 1

            while epoch <= n_epoch:

                if epoch != 1:
                    self.saver.restore(sess, self.path + self.model_name + ".ckpt")

                epoch_loss = 0
                total_batches = self.n_docs / float(batch_size)
                batch_x = []
                batch_y = []
                batches_run = 0
                time0 = datetime.now()

                for i, doc_vector in enumerate(train_inputs):
                    batch_x.append(list(doc_vector))
                    batch_y.append(list(train_labels[i]))

                    if batches_run % n_recorded_batch == 0 and batches_run != 0:
                        elapsed = (datetime.now() - time0)
                        second = elapsed.total_seconds()
                        time = time_format(second)
                        time_per_batch = float(second) / n_recorded_batch
                        time0 = datetime.now()
                        print('Batch run:', batches_run, '/', total_batches, '| Epoch:', epoch,
                              '| Average batch loss:', epoch_loss / (batches_run), '| Time:', time[0], 'hours',
                              time[1], 'minutes', "%.2f" % time[2], 'seconds.')
                        remaining_time = time_format(
                            (total_batches * time_per_batch * (n_epoch - epoch + 1)) - batches_run * time_per_batch)
                        print('Estimated remaining time:', remaining_time[0], 'hours', remaining_time[1], 'minutes',
                              "%.2f" % remaining_time[2], 'seconds.')

                    if len(batch_x) >= batch_size:
                        _, c = sess.run([optimizer, cost], feed_dict={self.x: batch_x, self.y: batch_y})

                        epoch_loss += c

                        batch_x = []
                        batch_y = []
                        batches_run += 1

                self.saver.save(sess, save_path=self.path + self.model_name + ".ckpt")
                print('Epoch', epoch, 'completed out of', n_epoch, 'loss:', epoch_loss)
                with open(self.log, 'wb') as f:
                    pickle.dump(epoch, f)
                epoch += 1

    def test_nn_bow(self, test_inputs, test_labels, category=None, threshold=None):
        prediction = self.create_model(self.x)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            try:
                self.saver.restore(sess, self.path + self.model_name + ".ckpt")
            except Exception as e:
                print(str(e))
                return

            y_p = tf.argmax(prediction, 1)
            correct = tf.equal(y_p, tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            feature_sets = []
            labels = []
            counter = 0

            for i, doc_vector in enumerate(test_inputs):
                try:
                    feature_sets.append(doc_vector)
                    labels.append(test_labels[i])
                    counter += 1
                except Exception as e:
                    print(str(e))

            print('Tested', counter, 'samples.')
            test_x = np.array(feature_sets)
            test_y = np.array(labels)

            val_accuracy, y_pred, pred = sess.run([accuracy, y_p, prediction], feed_dict={self.x: test_x, self.y: test_y})

            print("Validation accuracy:", val_accuracy)

            if category == "binary":
                if threshold is not None:
                    for i, p in enumerate(y_pred):
                        if sigmoid(max(pred[i]) - min(pred[i]), theta=0.2) < threshold:
                            y_pred[i] = 0
                y_true = np.argmax(test_labels, 1)
                y_true = 1 - y_true
                y_pred = 1 - y_pred

                if threshold is not None:
                    print("Applied threshold accuracy:", metrics.accuracy_score(y_true=y_true, y_pred=y_pred))
                print("Precision:", metrics.precision_score(y_true=y_true, y_pred=y_pred))
                print("Recall:", metrics.recall_score(y_true=y_true, y_pred=y_pred))
                print("F1 score:", metrics.f1_score(y_true=y_true, y_pred=y_pred))
                print("Confusion matrix:")
                print(metrics.confusion_matrix(y_true=y_true, y_pred=y_pred))

    def predict(self, new_docs, reverse_class_dict=None, top_n=3, category=None, threshold=None):
        prediction = self.create_model(self.x)
        predictions = list()

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            try:
                self.saver = tf.train.import_meta_graph(self.path + self.model_name + ".ckpt.meta")
                self.saver.restore(sess, self.path + self.model_name + ".ckpt")
            except Exception as e:
                print(str(e))
            for doc in new_docs:
                confidence = "N/A"
                bow_vector = create_bow_vector(lexicon=self.lexicon, doc=doc)
                bow_vector = rescale_and_normalize(self.lexicon, bow_vector, self.doc_counts, self.n_docs)
                top = np.array(prediction.eval(feed_dict={self.x: [bow_vector]}))
                if category == 'binary':
                    print(top)
                    confidence = sigmoid(max(top[0]) - min(top[0]), theta=0.2)
                    if threshold is not None and confidence < threshold:
                        predictions.append([["ED"], confidence])
                        continue
                top = top.argsort()
                top = top[0][::-1][:top_n]
                results = []
                for result in top:
                    pred = np.zeros(self.n_classes)
                    pred[result] = 1
                    pred = tuple(pred)
                    results.append(reverse_class_dict[pred].split(":")[0])
                predictions.append([results, confidence])
        tf.reset_default_graph()

        return predictions
