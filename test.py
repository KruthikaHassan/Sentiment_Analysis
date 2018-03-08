

import csv
import re
from vocab_vectors import build_vocab
from vocab_vectors import VocabVector

text_token_flags = re.MULTILINE | re.DOTALL

def cleanup(string):
    ''' Cleans up the given tweet '''

    def allcaps(txt):
        txt = txt.group()
        return txt.lower() + " <allcaps> "

    # function so code less repetitive
    def re_sub(pattern, repl):
        try:
            return re.sub(pattern, repl, string, flags=text_token_flags)
        except ValueError:
            return string

    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    string = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " <url> ")
    string = re_sub(r"@\w+", " <user> ")
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
    #string = re_sub(r"[^\<\>A-Za-z0-9]+", " ")
    return string.lower()



file = open('SAD.csv')
clean_file  = open('sad_tokens.txt', 'w')

data  = [row for row in csv.reader(file)]
ww = 0
for row in data[1:]:
    line = cleanup(row[-1])
    words = line.split()
    for word in words:
        clean_file.write("%s " % (word))
        ww += 1

print(ww)

clean_file.close()
file.close()


