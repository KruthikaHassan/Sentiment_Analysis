"""
Main function module
"""
#!/usr/bin/env python3

__author__ = "Kruthika H A"
__email__ = "kruthika@uw.edu"

import sys
import numpy as np
import random
from word_vectors import WordVector
from word_vectors import LoadWV
from data_set import DataSet
from data_set import LoadCSV
from sentiment_classifier import SentimentClassifier

class Configuration:
    def print(self):
        attrs = vars(self)
        print("Configuration:")
        for item in attrs:
            print("%s : %s" % (item, attrs[item]))

def split_dataset(dataset, train_percent=None):
    ''' Splits the dataset into train and test '''

    if not train_percent or int(train_percent) > 100:
        train_percent = 80
    
    print("Splitting Train:Test %d:%d %%" % (train_percent, 100-train_percent) )
        
    # Shuffle / Randamize the indecies
    data_indecies = [i for i in range(dataset.num_records)]
    random.shuffle(data_indecies)

    # How many traininig data we need? 
    num_train_records = int(train_percent) * dataset.num_records // 100

    # Init train and test 
    train_text, train_labels = [], []
    test_text, test_labels = [], []

    for index in data_indecies:
        if index < num_train_records:
            train_labels.append(dataset.labels[index])
            train_text.append(dataset.text[index])
        else:
            test_labels.append(dataset.labels[index])
            test_text.append(dataset.text[index])
    
    train_dataset = DataSet(train_text, train_labels, dataset.isVectorized, dataset.word_vector)
    test_dataset  = DataSet(test_text, test_labels, dataset.isVectorized, dataset.word_vector)

    return train_dataset, test_dataset

def main(data_file_path, word_vec_filename, batch_size=50, lstmUnits=24, epochs=10):
    """ Main function """

    # Raw dataset
    data_set = LoadCSV(data_file_path)
    #data_set.vectorize_text(word_vector=all_words_vector, normalized_length=50)
    #word_vector = data_set.word_vector

    # Global word_vector
    all_words_vector = LoadWV(word_vec_filename)

    # Train and test 
    train_dataset, test_dataset = split_dataset(data_set)

    train_dataset.vectorize_text(word_vector=all_words_vector, normalized_length=250)
    word_vector = train_dataset.word_vector

    test_dataset._word_vector = word_vector
    test_dataset.vectorize_text(word_vector=None, normalized_length=train_dataset.max_text_length)

    print("Train dataset numrecords: %d:" % (train_dataset.num_records))
    print("Test dataset numrecords: %d:" % (test_dataset.num_records))

    # Set some config params for this dataset
    config = Configuration()
    config.batchSize     =  batch_size
    config.lstmUnits     =  lstmUnits
    config.numClasses    =  train_dataset.num_classes
    config.maxSeqLength  =  train_dataset.max_text_length
    config.numDimensions =  word_vector.dimension
    config.print()

    # Init classifier
    classifier = SentimentClassifier(config, word_vector.embeddings)

    # Train
    for epoch_num in range(epochs):
        classifier.fit_epoch(train_dataset, epoch_num)
        train_accuracy = classifier.accuracy(train_dataset) * 100
        test_accuracy = classifier.accuracy(test_dataset) * 100
        print("%d:%.2f:%.2f" % (epoch_num, train_accuracy, test_accuracy), end='    ', flush=True)
        #print("Epoch Num: %d" % epoch_num)
        #print("Train Accuracy = %.2f %%" % (classifier.accuracy(train_dataset) * 100))
        #print("Test  Accuracy = %.2f %%" % (classifier.accuracy(test_dataset) * 100))

    print("")

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