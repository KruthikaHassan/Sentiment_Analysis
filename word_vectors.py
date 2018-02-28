
import sys
import time
import csv
import re
import numpy as np

class WordVector(object):
    def __init__(self, vocab, embeddings, dimension=None):
        self._vocab      = vocab
        self._embeddings = embeddings
        self._dimension  = len(embeddings[0])
        
    @property
    def vocab(self):
        return self._vocab
    
    @property
    def embeddings(self):
        return np.array(self._embeddings, dtype=np.float32)
    
    @property
    def dimension(self):
        return self._dimension

#########################################################################

class LoadWV(WordVector):

    def __init__(self, vocab_file_path):

        start_time = time.time()
        print("Loading File:", vocab_file_path)
        vocab, embeddings = self.__load_glove_vectors(vocab_file_path)
        time_taken = time.time() - start_time
        print("%s Loaded: %.3f secs!" % (vocab_file_path, time_taken))

        super().__init__(vocab, embeddings)
    
    @property
    def embeddings(self):
        return self._embeddings

    def __load_glove_vectors(self, filename):
        vocab = []
        embd = []
        with open(filename, 'r') as file:
            for line in file:
                row = line.strip().split(' ')
                dim = len(row[1:])
                isexpected_dim = (dim == 25) or (dim == 50) or (dim == 100) or (dim == 200)
                if not isexpected_dim:
                    continue
                vocab.append(row[0])
                embd.append([ float(s) for s in row[1:] ])
        return vocab, embd
