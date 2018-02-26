"""
Main function module
"""
#!/usr/bin/env python3

__author__ = "Kruthika H A"
__email__ = "kruthika@uw.edu"

import sys
from data_handler import WordVector
from data_handler import Dataset
from sentiment_classifier import SentimentClassifier

class Configuration:
    pass

def accuraccy():
    pass

def main(data_file_path, word_vec_filename, lstmUnits, iterations):
    """ Main function """

    

    # First Load Vocab
    word_vectors = WordVector(word_vec_filename)

    # Get data set ready
    data_set = Dataset(data_file_path, word_vectors)

    # Set some config params for this dataset
    config = Configuration()

    config.batchSize     =  data_set.num_records
    config.numClasses    =  data_set.num_classes
    config.maxSeqLength  =  data_set.max_text_length
    config.numDimensions =  len(word_vectors.embeddings[0])
    config.lstmUnits     =  lstmUnits
    config.iterations    =  iterations
    
    attrs = vars(config)
    print("Configuration:")
    for item in attrs:
        print("%s : %s" % (item, attrs[item]))

    classifier = SentimentClassifier(config, data_set.embeddings)

    classifier.fit(data_set, config.iterations)

    print("Accuracy = %.2f %%" % (classifier.accuracy(data_set) * 100))

    # accuraccy(exptected_labels, predicted_labes)

if __name__ == "__main__":
    ''' Start the program here '''

    if len(sys.argv) != 5:
        print("Usage: $python3 main.py train.csv glove_vec.txt lstmunits iterations")
        exit()

    # Parrse arguments 
    data_file_path = sys.argv[1]
    word_vec_filename = sys.argv[2]
    lstmUnits = int(sys.argv[3])
    iterations = int(sys.argv[4])
    
    # Run the program!
    main(data_file_path, word_vec_filename, lstmUnits, iterations)