import os
import sys
import csv
import json
import time
from nltk.tokenize import TweetTokenizer

import numpy as np
from keras.preprocessing import sequence
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime 
from collections import Counter
import pdb


import pandas as pd
from itertools import chain

try:
    import cPickle as pickle
except ImportError:
    import pickle

HOME = os.path.join(os.path.expanduser('~'), 'quora-duplicate-questions')
TRAIN = os.path.join(HOME, 'data', 'train.csv')
PROCESSED = os.path.join(HOME, 'processed_data')

def load():
    # read in data using csv
    data = pd.read_csv(TRAIN).loc[1:, :]
    X = data[['question1', 'question2']].astype(str)
    y = data['is_duplicate']
    return X, y

def tokenize(tkn, X):
    return (X['question1'].apply(tkn).values, X['question2'].apply(tkn).values)

def build_dataset(q1, q2, vocab_size):

    """ Re turn two lists, one for `question1` and another for `question2`
        Each list contain integers representation of tokens
        for each sentence. 

        [[ 1 , 56, 5, 9], [...] ] ,  [[20, 0, -1, 5, 2], [...]]
    """
    # get the most common words in the vocab
    # and build index with those words 
    # we'll replace words in not our vocab with the token 'UNK'

    
    count = [('UNK', -1)] 
    gathered = (x for sent in chain(q1, q2) for x in sent)
    count.extend(Counter(gathered).most_common(vocab_size -1))
    word_indx = {w : i+1 for i, (w, _) in enumerate(count)}
    # encode the data using this index
    q1_tok = [[word_indx.get(w) or -1 for w in sent] for sent in q1] 
    q2_tok = [[word_indx.get(w) or -1 for w in sent] for sent in q2]
    
    print("Sample vector from question1: {}\n\n".format(q1_tok[:2])) 
    print("Sample vector from question2: {}".format(q2_tok[:2])) 
    return q1_tok, q2_tok, word_indx 

if __name__ == '__main__':
    #  pad with 0's
    #  to make room for that padding, start word index from 1, see: `build_dataset`
    # 'UNK' = -1 :

    vocab_size = 50000
    maxlen = 50 # maxlen of the question encoding
    
    tkn = TweetTokenizer().tokenize
    enc = OneHotEncoder()
    
    print("Loading data from: ", TRAIN) 
    X, y = load()
    
    print("Tokenizing and Vectorizing questions...")
    q1_tok, q2_tok = tokenize(tkn, X)
    q1_vec, q2_vec, word_indx =  build_dataset(q1_tok, q2_tok, vocab_size=vocab_size)
    del X
    # padding with 0 to the left at maxlen width
    q1_pad = sequence.pad_sequences(q1_vec, maxlen)
    q2_pad = sequence.pad_sequences(q2_vec, maxlen)
    
    # one hot encode the target
    # we are going to have to reshape is_duplicate to be a colunmn vectora
    # print("One Hot Encoding the target...")
    _y = y.values.reshape(-1, 1)
    del y

    print("Sample of One Hot Encoded targets", _y[:10, :])
    
    # place these transformed vectoes in a time stamped folder
    day = datetime.now().strftime("%A_%d_%m_%_H_%-M")
    
    try:
        folder = os.path.join(PROCESSED, 'training', day)
        os.makedirs(folder)
    except FileExistError:
        pass

    print('Writing features and target to timestamped folder :', folder) 
    np.save(open(os.path.join(folder, 'q1_X_train_padded.npy') , 'wb'), q1_pad)
    np.save(open(os.path.join(folder, 'q2_X_train_padded.npy') , 'wb'), q2_pad)
    np.save(open(os.path.join(folder, 'y_train.npy'), 'wb'), _y)
    pickle.dump(word_indx, open(os.path.join(folder, 'word_index.pkl'),'wb'))

    print("Done!")
