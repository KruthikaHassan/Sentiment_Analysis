
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
        self._logits     = (tf.matmul(last, weight) + bias)
        self._y_p        = tf.argmax(self._logits,1)
        
        y_labels         = tf.argmax(self._label_placeholder,1)

        # Setup optimizer
        correctPred     = tf.equal(self._y_p, y_labels)
        self._accuracy  = tf.reduce_mean(tf.cast(correctPred, tf.float32))
        self._loss      = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self._logits, labels=self._label_placeholder))
        self._optimizer = tf.train.AdamOptimizer().minimize(self._loss)
        
        # Some useful metrics
        self._precision, self._prec_op = tf.metrics.precision(y_labels, self._y_p)
        self._recall, self._rec_op     = tf.metrics.recall(y_labels, self._y_p)

        # Session
        self._sess  = tf.Session()

        # Init global vars
        self._sess.run(tf.global_variables_initializer())     
        self._sess.run(tf.local_variables_initializer())
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
    
    def predict(self, test_data):
        sess           = self._sess
        prediction     = self._y_p
        input_data     = self._input_placeholder
        
        predictions = []
        test_data.reset_epoch()
        while not test_data.epoch_completed:
            nextBatch, _ = test_data.get_next_batch(self._batch_size)
            b_pred = sess.run(prediction , {input_data: nextBatch})
            predictions += list(b_pred)

        return predictions[0:test_data.num_records]

    def metrics(self, test_data):
        sess           = self._sess
        input_data     = self._input_placeholder
        labels         = self._label_placeholder
        precision      = self._precision
        recall         = self._recall
        accuracy       = self._accuracy
        loss           = self._loss

        prec_op        = self._prec_op
        rec_op         = self._rec_op

        accs        = []
        losses      = []

        test_data.reset_epoch()
        while not test_data.epoch_completed:
            nextBatch, nextBatchLabels = test_data.get_next_batch(self._batch_size)
            
            b_pred, b_loss, b_acc = sess.run((self._y_p, loss, accuracy), {input_data: nextBatch, labels:nextBatchLabels})
            sess.run([prec_op, rec_op], {labels:nextBatchLabels, self._y_p:b_pred})
            
            accs.append(b_acc)
            losses.append(b_loss)
        
        prec, rec = sess.run([precision, recall])
        f1s  = 2 * prec * rec / ( prec + rec )

        ret_vals = {
            'accuracy'    : np.average(accs),
            'precision'   : prec,
            'recall'      : rec,
            'f1_score'    : f1s,
            'loss'        : np.average(losses)
        }

        return ret_vals

        