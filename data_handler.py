
import sys
import time
import csv
import re
import numpy as np


class WordVector:

    def __init__(self, vocab_file_path):

        start_time = time.time()
        print("Loading File:", vocab_file_path)
        self._vocab, self._embeddings = self.load_glove_vectors(vocab_file_path)
        time_taken = time.time() - start_time
        print("%s Loaded: %.3f secs!" % (vocab_file_path, time_taken))

    @property
    def vocab(self):
        return self._vocab
    
    @property
    def embeddings(self):
        return self._embeddings

    def load_glove_vectors(self, filename):
        vocab = []
        embd = []
        file = open(filename,'r')
        for line in file.readlines():
            row = line.strip().split(' ')

            if len(row[1:]) != 25:
                continue

            vocab.append(row[0])
            embd.append([ float(s) for s in row[1:] ])
        file.close()
        return vocab, embd


class Dataset:

    def __init__(self, data_file_path, wordVectors):
        self.text_token_flags = re.MULTILINE | re.DOTALL

        self.all_vocab = wordVectors.vocab
        self.all_embeddings = wordVectors.embeddings

        self._used_vocab = []
        self._used_embeddigs = []

        start_time = time.time()
        print("Loading File:", data_file_path)
        self._text, self._labels = self.load_csv_file(data_file_path)
        time_taken = time.time() - start_time
        print("%s Loaded: %.3f secs!" % (data_file_path, time_taken))

        start_time = time.time()
        print("Vectorizing text:", len(self._text))
        self._vec_text = self.vectorize_text()
        time_taken = time.time() - start_time
        print("%d lines of text vectorized in %.3f secs!" % (len(self._vec_text), time_taken))

####################################### Useful properties #####################################

    @property
    def text(self):
        return self._vec_text 
    
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
        return len(self._vec_text[0])
    
    @property 
    def vocab(self):
        return self._used_vocab
    
    @property
    def embeddings(self):
        return np.array(self._used_embeddigs, dtype=np.float32)

###################################### Sending batch wise data ################################

    def split_data(self):
        # split data into test and train
        pass

    def get_next_train_batch(self, batch_size=None):

        # For now lets return everything
        return self._vec_text, self._labels

####################################### Loading from file ######################################
    
    def load_csv_file(self, filename):
        file = open(filename)
        data  = [row for row in csv.reader(file)]
        
        text, labels = [], []
        for row in data[1:101]:
            text.append(row[-1])
            labels.append(self.get_vect_label(row[1]))
        return text, labels

###################################### Data Conversion ##########################################


    def vocab_index(self, word):

        all_embeddings = self.all_embeddings
        all_vocab      = self.all_vocab

        if word not in self._used_vocab:
            if word in all_vocab:
                index = all_vocab.index(word)
                self._used_vocab.append(word)
                self._used_embeddigs.append(all_embeddings[index])

        return self._used_vocab.index(word)



    def get_vect_label(self, label_text):

        if label_text == 'negative':
            return [1, 0]
        else:
            return [0, 1]

        # if label_text == 'positive':
        #     return [1, 0, 0, 0]
        # elif label_text == 'negative':
        #     return [0, 1, 0, 0]
        # elif label_text == 'neutral':
        #     return [0, 0, 1, 0]
        # elif label_text == 'irrelevant':
        #     return [0, 0, 0, 1]

    def vectorize_text(self):
        start_time = time.time()
        unknown_word_index = self.vocab_index('<unknown>')
        text_vector = []
        words_without_vector = []
        line_num = 0
        
        max_line_length = 0
        for line in self._text:
            line_vec = []
            split_line = self.tokenize(line).split()
            for word in split_line:
                try:
                    word_vec = self.vocab_index(word)
                except ValueError:
                    if word not in words_without_vector:
                        words_without_vector.append(word)
                    word_vec = unknown_word_index
                line_vec.append(word_vec)
            text_vector.append(line_vec)
            
            if len(line_vec) > max_line_length:
                max_line_length = len(line_vec)

            line_num += 1
            if line_num % 100 == 0:
                print("Processing line: %d, time = %.2f" % (line_num, time.time()- start_time))

        # Normalize indicies
        total_lines = line_num
        normalized_text_vector = np.zeros((total_lines, max_line_length), dtype='int32')
        for line_num in range(total_lines):
            lineLen = len(text_vector[line_num])
            normalized_text_vector[line_num, 0:lineLen] = text_vector[line_num]
        
        print(max_line_length)
        #print("No vector words:", len(words_without_vector))
        return normalized_text_vector

######################################## Text Cleanup  ##############################################

    def hashtag(self, string):
        string = string.group()
        hashtag_body = string[1:]
        if hashtag_body.isupper():
            result = " {} ".format(hashtag_body.lower())
        else:
            result = " ".join([" <hashtag> "] + re.split(r"(?=[A-Z])", hashtag_body, flags=self.text_token_flags))
        return result

    def allcaps(self, string):
        string = string.group()
        return string.lower() + " <allcaps> "

    def tokenize(self, string):
        # Different regex parts for smiley faces
        eyes = r"[8:=;]"
        nose = r"['`\-]?"

        # function so code less repetitive
        def re_sub(pattern, repl):
            try:
                return re.sub(pattern, repl, string, flags=self.text_token_flags)
            except ValueError:
                return string

        string = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " <url> ")
        string = re_sub(r"@\w+", " <user>")
        #string = re_sub(r"#\S+", self.hashtag)
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

        ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
        # string = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
        string = re_sub(r"([A-Z]){2,}", self.allcaps)
        string = re.sub(r"[^\<\>A-Za-z0-9]+", " ", string.lower())

        return string

####################################################################################