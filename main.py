"""
Main function module
"""
#!/usr/bin/env python3

__author__ = "Kruthika H A"
__email__ = "kruthika@uw.edu"

import sys
import numpy as np
import random
from vocab_vectors import VocabVector
from vocab_vectors import build_vocab
from data_set import DataSet
from data_set import LoadCSV
from data_set import split_dataset
from sentiment_classifier import SentimentClassifier

class Configuration:
    def print(self):
        attrs = vars(self)
        print("Configuration:")
        for item in attrs:
            print("%s : %s" % (item, attrs[item]))



def main(data_file_path, word_vec_filename, batch_size=50, lstmUnits=24, epochs=10):
    """ Main function """

    # Raw dataset
    data_set = LoadCSV(data_file_path)
    #vocab_vector = build_vocab(data_set.text)
    #data_set.vectorize_text(vocab_vector.vocab, 50)

    # Train and test 
    train_dataset, test_dataset = split_dataset(data_set, 70)
    vocab_vector = build_vocab(train_dataset.text, word_vec_filename)
    train_dataset.vectorize_text(vocab_vector.vocab, 100)

    test_dataset.vectorize_text(vocab_vector.vocab, normalized_length=train_dataset.max_text_length)

    print("Train dataset numrecords: %d:" % (train_dataset.num_records))
    print("Test dataset numrecords: %d:" % (test_dataset.num_records))

    # Set some config params for this dataset
    config = Configuration()
    config.batchSize     =  batch_size
    config.lstmUnits     =  lstmUnits
    config.numClasses    =  train_dataset.num_classes
    config.maxSeqLength  =  train_dataset.max_text_length
    config.numDimensions =  vocab_vector.dimension
    config.print()

    # Init classifier
    classifier = SentimentClassifier(config, vocab_vector.embeddings)

    # Train
    train_accs = []
    test_accs  = []
    for epoch_num in range(epochs):
        classifier.fit_epoch(train_dataset, epoch_num)
        train_accuracy = classifier.accuracy(train_dataset) * 100
        test_accuracy = classifier.accuracy(test_dataset) * 100
        print("%d:%.2f:%.2f" % (epoch_num, train_accuracy, test_accuracy), end='    ', flush=True)
        train_accs.append(train_accuracy)
        test_accs.append(test_accuracy)
        #print("Epoch Num: %d" % epoch_num)
        #print("Train Accuracy = %.2f %%" % (classifier.accuracy(train_dataset) * 100))
        #print("Test  Accuracy = %.2f %%" % (classifier.accuracy(test_dataset) * 100))

    print("")

    print("Train accuracies:")
    print(train_accs)
    print("Test Accuracies:")
    print(test_accs)

if __name__ == "__main__":
    ''' Start the program here '''

    if len(sys.argv) != 6 :
        print("Usage: $python3 main.py train.csv glove_vec.txt lstmunits epochs batch_size")
        exit()
    
    data_file_path = sys.argv[1]
    word_vec_filename = sys.argv[2]    
    lstmUnits = int(sys.argv[3])
    epochs = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    
    # Run the program!
    main(data_file_path, word_vec_filename, batch_size, lstmUnits, epochs)