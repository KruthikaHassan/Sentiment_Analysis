import sys
import time
import csv
import re
import numpy as np
import random
from word_vectors import WordVector

class DataSet(object):

    def __init__(self, text, labels, isVectorized=False, word_vector=None):
        
        # General Init
        self._text         = text
        self._labels       = labels
        self._isVectorized = isVectorized
        self._word_vector  = word_vector

        # For batchwise data retrival
        self._record_indecies  =   [i for i in range(self.num_records)]
        

    #####################  Properties #################################

    @property
    def text(self):
        return self._text 
    
    @property
    def labels(self):
        return self._labels

    @property
    def num_records(self):
        return len(self._labels)
    
    @property
    def num_classes(self):
        return len(self._labels[0])
    
    @property 
    def max_text_length(self):
        return len(self._text[0])
    
    @property
    def isVectorized(self):
        return self._isVectorized

    @property 
    def word_vector(self):
        return self._word_vector
    
    @property
    def epoch_completed(self):
        return self._epoch_completed
    
    @property
    def records_used(self):
        return self._records_used

    ######################### Methods ##################################

    def reset_epoch(self):
        random.shuffle(self._record_indecies)
        self._epoch_completed  = False
        self._records_used     = 0
    
    def get_next_batch(self, batch_size=None):
        ''' Gives data in batches '''
        
        if not batch_size:
            return self._text, self._labels
        
        text = []
        labels = []
        records_retrived = 0
        for index in self._record_indecies[self._records_used:]:
            text.append(self.text[index])
            labels.append(self.labels[index])
            records_retrived += 1
            if records_retrived >= batch_size:
                break
        
        self._records_used += records_retrived

        if records_retrived < batch_size:
            self._records_used       = 0
            self._epoch_completed    = True
            extra_records_needed     = batch_size-records_retrived
            extra_text, extra_labels = self.get_next_batch(extra_records_needed)
            text                    += extra_text
            labels                  += extra_labels

        return text, labels

    def vectorize_text(self, word_vector=None, normalized_length=100):
        ''' vectorize the cleaned up text '''
        
        # Already vectorized, no need to continue further
        if self.isVectorized:
            return True

        # Use this class' word, vector means, dont build one
        if not word_vector:
            build_word_vector = False
            all_vocab         = []
            all_embeddings    = []
            used_vocab        = self._word_vector.vocab
            used_embeddigs    = self._word_vector.embeddings
        else:
            build_word_vector = True
            all_vocab         = word_vector.vocab
            all_embeddings    = word_vector.embeddings
            used_vocab        = []
            used_embeddigs    = []
            
        # Function to avoid repetition
        def vocab_index(word):
            if (build_word_vector) and (word not in used_vocab) and (word in all_vocab):
                index = all_vocab.index(word)
                used_vocab.append(word)
                used_embeddigs.append(all_embeddings[index])
            return used_vocab.index(word)

        
        # Now lets indexise the words
        print("Vectorizing text: %d lines" % (len(self._text)))
        start_time = time.time()
        
        raw_text           = self._text
        text_vec           = []
        max_line_length    = 0
        line_num           = 0
        unknown_word_index = vocab_index('<unknown>')
        for line in raw_text:
            line_vec = []
            split_line = line.split()
            for word in split_line:
                try:
                    word_vec = vocab_index(word)
                except ValueError:
                    word_vec = unknown_word_index
                line_vec.append(word_vec)
            text_vec.append(line_vec)
            
            # Calculating max word in a line
            if len(line_vec) > max_line_length:
                max_line_length = len(line_vec)

            # Print a dot every 100 lines
            line_num += 1
            if line_num % 100 == 0:
                print('.', end='', flush=True)
        
        # Make all lines same length as max + buffer ( append zeroes to the end )
        if normalized_length < max_line_length:
            normalized_length = max_line_length
        total_lines     = line_num
        self._text      = np.zeros((total_lines, normalized_length), dtype='int32')
        for line_num in range(total_lines):
            lineLen = len(text_vec[line_num])
            if lineLen < normalized_length:
                self._text[line_num, 0:lineLen] = text_vec[line_num]
        
        time_taken = time.time() - start_time
        print("\n %d lines of text vectorized in %.3f secs!" % (len(self._text), time_taken))

        if build_word_vector:
            self._word_vector = WordVector(used_vocab, used_embeddigs)

        # Set vectorized as true
        self._isVectorized = True
    
#################################################### Child class to laod data from file #######################################################
    
class LoadCSV(DataSet):

    def __init__ (self, data_file_path):
        self.text_token_flags = re.MULTILINE | re.DOTALL

        start_time = time.time()
        print("Loading File:", data_file_path)
        text, labels = self.load_csv_file(data_file_path)
        time_taken = time.time() - start_time
        print("%s Loaded: %.3f secs!" % (data_file_path, time_taken))
        
        super().__init__(text, labels)

    def load_csv_file(self, filename):
        file = open(filename)
        data  = [row for row in csv.reader(file)]
        
        text, labels = [], []
        for row in data[1:]:
            line = self.cleanup(row[1])
            text.append(line)
            labels.append(self.get_vect_label(row[2:]))
        return text, labels

    def get_vect_label(self, label_text):

        # if label_text == 'negative':
        #     return [1, 0]
        # else:
        #     return [0, 1]

        # if label_text == 'positive':
        #     return [1, 0, 0, 0]
        # elif label_text == 'negative':
        #     return [0, 1, 0, 0]
        # elif label_text == 'neutral':
        #     return [0, 0, 1, 0]
        # elif label_text == 'irrelevant':
        #     return [0, 0, 0, 1]


        # "toxic","severe_toxic","obscene","threat","insult","identity_hate"
        return [int(s) for s in label_text]

    def cleanup(self, string):
        ''' Cleans up the given tweet '''
        def hashtag(txt):
            txt = txt.group()
            hashtag_body = txt[1:]
            if hashtag_body.isupper():
                result = " {} ".format(hashtag_body.lower())
            else:
                result = " ".join([" <hashtag> "] + re.split(r"(?=[A-Z])", hashtag_body, flags=self.text_token_flags))
            return result
        
        def allcaps(txt):
            txt = txt.group()
            return txt.lower() + " <allcaps> "

        # function so code less repetitive
        def re_sub(pattern, repl):
            try:
                return re.sub(pattern, repl, string, flags=self.text_token_flags)
            except ValueError:
                return string

        # Different regex parts for smiley faces
        eyes = r"[8:=;]"
        nose = r"['`\-]?"

        string = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " <url> ")
        string = re_sub(r"@\w+", " <user>")
        #string = re_sub(r"#\S+", hashtag)
        string = re_sub(r"#\S+", " <hashtag> ")
        string = re_sub(r"/"," / ")
        string = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), " <smile> ")
        string = re_sub(r"{}{}p+".format(eyes, nose), " <lolface> ")
        string = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), " <sadface> ")
        string = re_sub(r"{}{}[\/|l*]".format(eyes, nose), " <neutralface> ")
        string = re_sub(r"<3"," <heart> ")
        string = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", " <number> ")
        string = re_sub(r"([!?.]){2,}", r"\1 <repeat> ")
        string = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong> ")
        # string = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
        string = re_sub(r"([A-Z]){2,}", allcaps)
        string = re.sub(r"[^\<\>A-Za-z0-9]+", " ", string.lower())
        return string