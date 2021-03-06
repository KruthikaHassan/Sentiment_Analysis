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
    val_set_save_file      = 'val_dataset.pkl'
    test_set_save_file     = 'test_dataset.pkl'
    vocab_vector_save_file = 'vocab_vector.pkl'

    if saved:
        start_time = time.time()
        print("Loading Train :", train_set_save_file)
        train_dataset = load_obj(train_set_save_file)
        time_taken = time.time() - start_time
        print("%s Loaded: %.3f secs!" % (train_set_save_file, time_taken))

        start_time = time.time()
        print("Loading Validation :", vocab_vector_save_file)
        val_dataset  = load_obj(val_set_save_file)
        time_taken = time.time() - start_time
        print("%s Loaded: %.3f secs!" % (val_set_save_file, time_taken))

        start_time = time.time()
        print("Loading Test :", vocab_vector_save_file)
        vocab_vector  = load_obj(vocab_vector_save_file)
        time_taken = time.time() - start_time
        print("%s Loaded: %.3f secs!" % (vocab_vector_save_file, time_taken))

        start_time = time.time()
        print("Loading Test :", test_set_save_file)
        test_dataset  = load_obj(test_set_save_file)
        time_taken = time.time() - start_time
        print("%s Loaded: %.3f secs!" % (test_set_save_file, time_taken))
    else:
        # Raw dataset
        data_set = LoadCSV(data_file_path)

        # Train and test 
        train_dataset, test_dataset = split_dataset(data_set, 85)
        train_dataset, val_dataset  = split_dataset(train_dataset, 80)
        vocab_vector = build_vocab(train_dataset.text, word_vec_filename)

        train_dataset.vectorize_text(vocab_vector.vocab, 100)
        val_dataset.vectorize_text(vocab_vector.vocab, normalized_length=train_dataset.max_text_length)
        test_dataset.vectorize_text(vocab_vector.vocab, normalized_length=train_dataset.max_text_length)

        save_obj(train_dataset, train_set_save_file)
        save_obj(val_dataset, val_set_save_file)
        save_obj(test_dataset, test_set_save_file)
        save_obj(vocab_vector, vocab_vector_save_file)

    print("Train dataset numrecords: %d:" % (train_dataset.num_records))
    print("Validation dataset numrecords: %d:" % (val_dataset.num_records))
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
    
    val_acc_file = open('val_accs.txt', 'w')
    train_acc_file  = open('train_accs.txt', 'w')
    
    val_lss_file = open('val_loss.txt', 'w')
    train_lss_file  = open('train_loss.txt', 'w')
    
    for epoch_num in range(epochs):
        classifier.fit_epoch(train_dataset)
        
        t_m = classifier.metrics(train_dataset)
        v_m = classifier.metrics(val_dataset)

        train_accuracy =  t_m['accuracy'] * 100
        val_accuracy   =  v_m['accuracy'] * 100

        train_loss     = t_m['loss'] * 100
        val_loss       =  v_m['loss'] * 100

        
        print("%d  %.2f||%.2f  %.2f||%.2f" % (epoch_num, train_accuracy, val_accuracy, train_loss, val_loss), end='\n', flush=True)
        
        
        train_acc_file.write("%f, " % (train_accuracy))
        val_acc_file.write("%f, " % (val_accuracy))

        train_lss_file.write("%f, " % (train_loss))
        val_lss_file.write("%f, " % (val_loss))
        
        max_indx = np.argmax(val_accs)
        if val_accuracy > val_accs[max_indx]:
            val_accs[max_indx] = val_accuracy
        else:
            min_indx = np.argmin(val_accs)
            if val_accuracy > val_accs[min_indx]:
                val_accs[min_indx] = val_accuracy
            elif val_accuracy < 77.0: # Try to get upto desired accuracy
                print(".", end=' ')
            else:
                print("\nTerminating training:", val_accs)
                break

    print("")

    val_metrics = classifier.metrics(val_dataset)
    print("validation Status: \n", val_metrics)
    
    train_metrics = classifier.metrics(train_dataset)
    print("Train Status: \n", train_metrics)
    
    test_metrics = classifier.metrics(test_dataset)
    print("Test Status: \n", test_metrics)

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
    main(data_file_path, word_vec_filename, True, batch_size, lstmUnits, epochs)
