
import sys
import time
import csv
import re
import numpy as np

class VocabVector(object):
    def __init__(self, vocab, embeddings):
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
    
    @property
    def num_records(self):
        return len(self._vocab)

#########################################################################################################

def __load_glove_vectors(filename, sorted_words):

    start_time = time.time()
    print("  Loading GLOVE file: ", filename)
    vocab = []
    embd = []
    with open(filename, 'r') as file:
        for line in file:
            row = line.strip().split(' ')
            dim = len(row[1:])
            isexpected_dim = (dim == 25) or (dim == 50) or (dim == 100) or (dim == 200)
            
            word = row[0]
            if (not isexpected_dim) or (word not in sorted_words):
                continue

            vocab.append(word)
            embd.append([ float(s) for s in row[1:] ])

            print('.', end='', flush=True)
    
    time_taken = time.time() - start_time
    print("\n  File loaded: %.3f secs!" % (time_taken))
    return VocabVector(vocab, embd)

def build_vocab(text, glove_file=None, threshold=5):

    start_time = time.time()
    print("Building vocab")

    all_vocab = {}
    for line in text:
        for word in line.split():
            if word not in all_vocab:
                all_vocab[word] = 1
            else:
                all_vocab[word] += 1

    # add unknown, start, end to required list
    required_vocab = {'<unknown>' : 0, '<MASK>' : 1000000000}
    for word in all_vocab:
        val = all_vocab[word]
        if val >= threshold:
            required_vocab[word] = val
        else:
            required_vocab['<unknown>'] += 1
    
    # Sort according to highest 
    sorted_words = sorted(required_vocab, key=lambda k: required_vocab[k], reverse=True)

    if glove_file:
        vocab_vector = __load_glove_vectors(glove_file, sorted_words)
        vocab_vector._vocab.insert(0, '<MASK>')
        vocab_vector._embeddings.insert(0, [0.0 for i in range(vocab_vector.dimension)])
    else:
        # binary embeddings
        num_words = len(sorted_words)
        max_bits  = len(list(bin(num_words))[2:])
        vocab, embds = ['<MASK>'], [[0 for i in range(max_bits)]]
        for word in sorted_words[1:]:
            bin_embd  = [0 for i in range(max_bits)]
            
            bits = list(bin(num_words))[2:]
            bits_len = len(bits)
            bin_embd[-bits_len:] = [int(i) for i in bits]

            vocab.append(word)
            embds.append(bin_embd)
            num_words -= 1
            print('.', end='', flush=True)
        
        vocab_vector = VocabVector(vocab, embds)

    time_taken = time.time() - start_time
    print("\nVocabulary built: %.3f secs!" % (time_taken))

    print("Total words:", vocab_vector.num_records)
    print("Words not included:", required_vocab['<unknown>'])

    return vocab_vector