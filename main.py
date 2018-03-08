"""
Main function module
"""
#!/usr/bin/env python3

__author__ = "Kruthika H A"
__email__ = "kruthika@uw.edu"

import sys
import numpy as np
import random
import pickle
import time
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

def save_obj(obj, filename):
    ''' Saves the dataset so we can load it next time '''
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_obj(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj


def main(data_file_path, word_vec_filename, saved=True, batch_size=50, lstmUnits=24, epochs=10):
    """ Main function """

    train_set_save_file    = 'train_dataset.pkl'
    test_set_save_file     = 'test_dataset.pkl'
    vocab_vector_save_file = 'vocab_vector.pkl'

    if saved:
        start_time = time.time()
        print("Loading Train :", train_set_save_file)
        train_dataset = load_obj(train_set_save_file)
        time_taken = time.time() - start_time
        print("%s Loaded: %.3f secs!" % (train_set_save_file, time_taken))

        start_time = time.time()
        print("Loading Test :", vocab_vector_save_file)
        vocab_vector  = load_obj(vocab_vector_save_file)
        time_taken = time.time() - start_time
        print("%s Loaded: %.3f secs!" % (vocab_vector_save_file, time_taken))

        start_time = time.time()
        print("Loading Vocab :", test_set_save_file)
        test_dataset  = load_obj(test_set_save_file)
        time_taken = time.time() - start_time
        print("%s Loaded: %.3f secs!" % (test_set_save_file, time_taken))
    else:
        # Raw dataset
        data_set = LoadCSV(data_file_path)

        # Train and test 
        train_dataset, test_dataset = split_dataset(data_set, 70)
        vocab_vector = build_vocab(train_dataset.text, word_vec_filename)

        train_dataset.vectorize_text(vocab_vector.vocab, 100)
        test_dataset.vectorize_text(vocab_vector.vocab, normalized_length=train_dataset.max_text_length)

        save_obj(train_dataset, train_set_save_file)
        save_obj(test_dataset, test_set_save_file)
        save_obj(vocab_vector, vocab_vector_save_file)

    print("Train dataset numrecords: %d:" % (train_dataset.num_records))
    print("Test dataset numrecords: %d:" % (test_dataset.num_records))

    # Set some config params for this dataset
    config = Configuration()
    config.epochs        =  epochs
    config.batchSize     =  batch_size
    config.lstmUnits     =  lstmUnits
    config.numClasses    =  train_dataset.num_classes
    config.maxSeqLength  =  train_dataset.max_text_length
    config.numDimensions =  vocab_vector.dimension
    config.print()

    # Init classifier
    classifier = SentimentClassifier(config, vocab_vector.embeddings)

    # Train
    val_accs = [0 for i in range(10)]
    train_accs = []
    test_accs  = []
    for epoch_num in range(epochs):
        classifier.fit_epoch(train_dataset)
        
        train_accuracy = classifier.accuracy(train_dataset) * 100
        test_accuracy = classifier.accuracy(test_dataset) * 100
        print("%d:%.2f:%.2f" % (epoch_num, train_accuracy, test_accuracy), end=' ', flush=True)
        train_accs.append(train_accuracy)
        test_accs.append(test_accuracy)
        
        max_indx = np.argmax(val_accs)
        if test_accuracy > val_accs[max_indx]:
            val_accs[max_indx] = test_accuracy
        else:
            min_indx = np.argmin(val_accs)
            if test_accuracy > val_accs[min_indx]:
                val_accs[min_indx] = test_accuracy
            elif test_accuracy < 85.0: # Try to get upto desired accuracy
                print(".", end=' ')
            else:
                print("\nTerminating training:", val_accs)
                break

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
    main(data_file_path, word_vec_filename, False, batch_size, lstmUnits, epochs)