
import tensorflow as tf
import datetime
import time
import numpy as np


class SentimentClassifier:

    def __init__(self, config, WordVectorEmbeddings):
        start_time = time.time()
        print("Initializing Classifier")
        
        tf.reset_default_graph()

        self._batch_size = config.batchSize
        
        # Place holders
        self._label_placeholder = tf.placeholder(tf.float32, [self._batch_size, config.numClasses])
        self._input_placeholder = tf.placeholder(tf.int32, [self._batch_size, config.maxSeqLength])

        # For the data
        #data = tf.Variable(tf.zeros([self._batch_size, config.maxSeqLength, config.numDimensions]),dtype=tf.float32)
        data = tf.nn.embedding_lookup(WordVectorEmbeddings, self._input_placeholder)
       
        # LSTM
        basic_lstmCell = tf.contrib.rnn.BasicLSTMCell(config.lstmUnits)
        lstmCell       = tf.contrib.rnn.DropoutWrapper(cell=basic_lstmCell, output_keep_prob=0.75)
        value, _       = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
        
        # Prediction
        weight           = tf.Variable(tf.truncated_normal([config.lstmUnits, config.numClasses]))
        bias             = tf.Variable(tf.constant(0.1, shape=[config.numClasses]))
        value            = tf.transpose(value, [1, 0, 2])
        last             = tf.gather(value, int(value.get_shape()[0]) - 1)
        self._prediction = (tf.matmul(last, weight) + bias)
        
        # Setup optimizer
        correctPred     = tf.equal(tf.argmax(self._prediction,1), tf.argmax(self._label_placeholder,1))
        self._accuracy  = tf.reduce_mean(tf.cast(correctPred, tf.float32))
        loss            = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self._prediction, labels=self._label_placeholder))
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        
        # Session
        self._sess  = tf.InteractiveSession()
        self._saver = tf.train.Saver()

        # Init global vars
        self._sess.run(tf.global_variables_initializer())     

        time_taken = time.time() - start_time
        print("Classifier Initialized: %.3f secs!" % (time_taken))

    def fit_epoch(self, train_data):
        
        # Retrive saved vars
        sess           = self._sess
        optimizer      = self._optimizer
        input_data     = self._input_placeholder
        labels         = self._label_placeholder
        
        train_data.reset_epoch()
        while not train_data.epoch_completed:
            nextBatch, nextBatchLabels = train_data.get_next_batch(self._batch_size)
            sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})
            #print('.', end='', flush=True)
       
        # print("\nTraining completed: %.3f secs!" % (time_taken))
        
        return None

    def accuracy(self, test_data):
        sess           = self._sess
        accuracy       = self._accuracy
        input_data     = self._input_placeholder
        labels         = self._label_placeholder
        
        test_data.reset_epoch()
        batch_accs = []
        while not test_data.epoch_completed:
            nextBatch, nextBatchLabels = test_data.get_next_batch(self._batch_size)
            acc = sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})
            batch_accs.append(acc)
        return np.average(batch_accs)
        