
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
        self._label_placeholder = tf.placeholder(tf.float32, [config.batchSize, config.numClasses])
        self._input_placeholder = tf.placeholder(tf.int32, [config.batchSize, config.maxSeqLength])

        # For the data
        data = tf.Variable(tf.zeros([config.batchSize, config.maxSeqLength, config.numDimensions]),dtype=tf.float32)
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

        # Setup summary log
        logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        tf.summary.scalar('Loss', loss)
        tf.summary.scalar('Accuracy', self._accuracy)
        self._merged_summary = tf.summary.merge_all()
        self._writer         = tf.summary.FileWriter(logdir, self._sess.graph)   

        # Init global vars
        self._sess.run(tf.global_variables_initializer())     

        time_taken = time.time() - start_time
        print("Classifier Initialized: %.3f secs!" % (time_taken))

    def fit_epoch(self, train_data, epoch_num):
        
        # Retrive saved vars
        sess           = self._sess
        optimizer      = self._optimizer
        merged_summary = self._merged_summary
        writer         = self._writer
        input_data     = self._input_placeholder
        labels         = self._label_placeholder
        saver          = self._saver        
        
        train_data.reset_epoch()
        while not train_data.epoch_completed:
            nextBatch, nextBatchLabels = train_data.get_next_batch(self._batch_size)
            sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})
        #    print('.', end='', flush=True) # to make sure something is happening
           
        #     #Write summary to Tensorboard
        #     if (i % 50 == 0):
        #         summary = sess.run(merged_summary, {input_data: nextBatch, labels: nextBatchLabels})
        #         writer.add_summary(summary, i)

        #      #Save the network every 10,000 training iterations
        #     if (i % 100 == 0 and i != 0):
        #         save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
        #         print("\nsaved to %s" % save_path)
        # writer.close()

        # time_taken = time.time() - start_time
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
        