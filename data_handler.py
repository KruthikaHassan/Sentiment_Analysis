
import numpy as np
from data_set import DataSet 

text_token_flags = re.MULTILINE | re.DOTALL

def cleanup(string):
    ''' Cleans up the given tweet '''
    def hashtag(txt):
        txt = txt.group()
        hashtag_body = txt[1:]
        if hashtag_body.isupper():
            result = " {} ".format(hashtag_body.lower())
        else:
            result = " ".join([" <hashtag> "] + re.split(r"(?=[A-Z])", hashtag_body, flags=text_token_flags))
        return result
    
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

def split_dataset(dataset, train_percent=None):
    ''' Splits the dataset into train and test '''

    if not train_percent or int(train_percent) > 100:
        print("Train percent Invalid, using default")
        train_percent = 80

    # Shuffle / Randamize the indecies
    data_indecies = [i for i in rage(dataset.num_records)]
    shuffled_indecies = np.shuffe(data_indecies)

    # How many traininig data we need? 
    num_train_records = int(train_percent) * dataset.num_records // 100

    # Init train and test 
    train_text, train_labels = [], []
    test_text, test_labels = [], []

    for index in shuffled_indecies:
        if index < num_train_records:
            train_labels.append(dataset.labels[index])
            train_text.append(dataset.text[index])
        else:
            test_labels.append(dataset.labels[index])
            test_text.append(dataset.text[index])
    
    train_dataset = DataSet(None, train_text, train_labels, dataset.isVectorized)
    test_dataset  = DataSet(None, test_text, test_labels, dataset.isVectorized)

    return train_dataset, test_dataset
    


    





