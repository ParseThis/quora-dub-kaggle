import os00
import sys
import csv
import json
import time
from nltk.tokenize import TweetTokenizer

import numpy as np
from keras.processing import sequence
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime 

try:
    import cPickle as pickle
except ModuleImportError:
    import pickle


PROCESSED = os.path.join(os.path.expanduser('~'), 'processed')

def load():
    # read in data using csv
    data = pd.read_csv(TRAIN)[1:, :]
    X = data[['question1', 'question2']].astype(str)
    y = data['is_duplicate'].astype(str)
    return X, y

def tokenize(tkn, X)
    return (X['q1'].apply(tkn).values, X['q2'].apply(tkn).values)

    

def vectorize(X_tok):
    words = set(s for sent in X_tok for x in sent)
    word_indx = {e : i+1 for (i, e) in enumerate(words)}
    indx_word = {i+1:  e for (i, e) in enumerate(words)}
    return words, word_indx


if __name__ == '__main__':
    maxlen = 50

    tkn = TweetTokenizer().tokenize
    enc = OneHotEncoder()
    X, y = load()
    q1_tok, q2_tok = tokenize(tkn, X)

    X_vec, word_indx =  vectorize(q1_tok + q2_tok)

    # create_vectors
    q1_vec  = [[word_indx[w] for w in sent] for sent in q1_tok]
    q2_vec  = [[word_indx[w] for w in sent] for sent in q2_tok]
    
    # padding with 0 to the left at maxlen width

    q1_pad = sequence.pad_sequence(q1_vex, maxlen)
    q2_pad = sequence.pad_sequence(q2_vex, maxlen)

    _y = enc.fit_transform(data['is_dub'].values.reshape(-1, 1)).toarray()
    
    now = datetime.now().isoformat()
    try:
        folder = os.path.join(PROCESSED, 'training', now)
        makedirs(folder)
    except FileExitError:
        pass

    np.save(q1_pad, open(os.path.join(folder, 'q1_X_train_padded.npy') , 'wb'))
    np.save(q2_pad, open(os.path.join(folder, 'q2_X_train_padded.npy') , 'wb'))
    np.save(_y, open(os.path.join(folder, 'y_train.npy'), 'wb'))

    pickle.dump(word_indx, open(os.path.join(folder, 'word_index.pkl'), 'wb')

