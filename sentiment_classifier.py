
import tensorflow as tf
import datetime


class SentimentClassifier:

    def __init__(self, config, WordVectorEmbeddings):
        tf.reset_default_graph()
        
        # Place holders
        self._labels = tf.placeholder(tf.float32, [config.batchSize, config.numClasses])
        self._input_data = tf.placeholder(tf.int32, [config.batchSize, config.maxSeqLength])

        # For the data
        data = tf.Variable(tf.zeros([config.batchSize, config.maxSeqLength, config.numDimensions]),dtype=tf.float32)

        print("Data1 Setup  Done")

        print(WordVectorEmbeddings.shape)

        data = tf.nn.embedding_lookup(WordVectorEmbeddings, self._input_data)
        
        print("Data Setup Done")

        # LSTM
        basic_lstmCell = tf.contrib.rnn.BasicLSTMCell(config.lstmUnits)
        lstmCell       = tf.contrib.rnn.DropoutWrapper(cell=basic_lstmCell, output_keep_prob=0.75)
        value, _       = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
        
        print("LSTM Setup Done")

        # Prediction
        weight           = tf.Variable(tf.truncated_normal([config.lstmUnits, config.numClasses]))
        bias             = tf.Variable(tf.constant(0.1, shape=[config.numClasses]))
        value            = tf.transpose(value, [1, 0, 2])
        last             = tf.gather(value, int(value.get_shape()[0]) - 1)
        self._prediction = (tf.matmul(last, weight) + bias)
        

        print("Prediction Setup Done")

        # Setup optimizer
        correctPred     = tf.equal(tf.argmax(self._prediction,1), tf.argmax(self._labels,1))
        accuracy        = tf.reduce_mean(tf.cast(correctPred, tf.float32))
        loss            = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self._prediction, labels=self._labels))
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        
        print("Optimizer Setup Done")

        # Session
        self._sess  = tf.InteractiveSession()
        self._saver = tf.train.Saver()

        print("Session started")

        # Setup summary log
        logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        tf.summary.scalar('Loss', loss)
        tf.summary.scalar('Accuracy', accuracy)
        self._merged_summary = tf.summary.merge_all()
        self._writer         = tf.summary.FileWriter(logdir, self._sess.graph)

        print("Summary Setup Done")

        
   

    def fit(self, train_data, iterations):
        
        # Retrive saved vars
        sess           = self._sess
        optimizer      = self._optimizer
        merged_summary = self._merged_summary
        writer         = self._writer
        input_data     = self._input_data
        labels         = self._labels
        saver          = self._saver        
        
        # Init global vars
        sess.run(tf.global_variables_initializer())

        for i in range(iterations):
           nextBatch, nextBatchLabels = train_data.get_next_train_batch()
           sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

           #Write summary to Tensorboard
           if (i % 50 == 0):
               summary = sess.run(merged_summary, {input_data: nextBatch, labels: nextBatchLabels})
               writer.add_summary(summary, i)

           #Save the network every 10,000 training iterations
           if (i % 10000 == 0 and i != 0):
               save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
               print("saved to %s" % save_path)
        writer.close()
        
        return None
    
    def predict(self):
        pass