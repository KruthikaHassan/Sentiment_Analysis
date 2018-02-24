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

def main(data_file_path, word_vec_filename):
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
    
    config.lstmUnits     =  12
    config.iterations    =  100
    
    print(config.__dict__)

    classifier = SentimentClassifier(config, data_set.embeddings)

    print(classifier.__dict__)
    # classifier.fit(data_set.traini, la)

    # predictions = classifier.predict(data_set.data, labels)

    # accuraccy(exptected_labels, predicted_labes)

if __name__ == "__main__":
    ''' Start the program here '''

    if len(sys.argv) != 3:
        print("Please provide Training and Test files to work with!")
        print("Usage: $python3 main.py train.csv")
        exit()

    # Parrse arguments 
    data_file_path = sys.argv[1]
    word_vec_filename = sys.argv[2]
    
    # Run the program!
    main(data_file_path, word_vec_filename)