import os.path
import sys
import pdb
import json
import pickle
from datetime import datetime
from zipfile import ZipFile
import numpy as np
from keras import backend as K
from keras.models import Model
import keras.layers
from keras.layers import    (LSTM, Merge, merge, Embedding, Dropout, Input, Dense,
                            BatchNormalization)
import keras.callbacks
import argparse

HOME = os.path.join(os.path.expanduser('~'), "quora-duplicate-questions")
MODELS = os.path.join(HOME, 'models')
LOGS = os.path.join(HOME, 'logs')
ZIPFILE = os.path.join(HOME, 'data', 'glove.42B.300d.zip')
GLOVEFILE = os.path.join(HOME,  'data', 'glove', 'glove.42B.300d.txt')
GLOVEZIP = 'glove.42B.300d.zip'
MAX_NB_WORDS = 100000
MAXLEN = 20
EMBEDDING_DIM = 300

class ShowPredictions(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        print(self.model.predict(batch, batch_size=len(batch)))


def makeEM(mx_words, word_index):

    embedding  = {}
    with open(GLOVEFILE) as g:
        for i, line in enumerate(g.readlines()):
            v = line.split(' ') # py3 syntax unpacking
            w = v[0]
            coeff = np.asarray(v[1:], dtype='float32')
            embedding[w] = coeff
    
    mx = min(len(embedding), mx_words)
    mat = np.zeros((mx + 2, EMBEDDING_DIM), dtype='float32')

    # map every word in my vocabulary
    for word, i in word_index.items():
        vec = embedding.get(word)
        if i > mx: # if the word index is greatner the numver  
            continue
        if vec is not None:
            mat[i] = vec

    return mat
                
def load_data():

    q1_X_train = np.load(TRAIN_Q1)           
    q2_X_train = np.load(TRAIN_Q2)           
    y_train = np.load(TRAIN_Y)
    
    with open(WORD_DIC, 'rb') as w:
        word_index = pickle.load(w)

    return q1_X_train, q2_X_train, y_train, word_index

def shared_lstm(embedding_matrix=None,
                maxlen=None,
                vocab_size=None,
                hidden_lstm=None,
                batch_size=None,
                dropout_lstm=None,
                dropout_embedding=None):
                
    q1_input = Input(shape=(maxlen,), name='q1_input')
    q2_input = Input(shape=(maxlen,), name='q2_input')
    
    q1_embed = Embedding(vocab_size+2, 
                        EMBEDDING_DIM, 
                        input_length=maxlen,
                        weights=[embedding_matrix],
                        trainable=False,
                        mask_zero=True,
                        name="q1_embedding")(q1_input)

    q2_embed = Embedding(vocab_size+2,
                        EMBEDDING_DIM,
                        input_length=maxlen,
                        weights=[embedding_matrix],
                        trainable=False,
                        mask_zero=True,
                        name="q2_embedding")(q2_input)


    lstm = LSTM(512, name="shared_lstm")
    q1_shared_lstm = lstm(q1_embed)
    q2_shared_lstm = lstm(q2_embed)

    gathered = keras.layers.concatenate([q1_shared_lstm, q2_shared_lstm], axis=-1)
    norm0 = BatchNormalization()(gathered)
    d1 = Dense(32, activation='relu', name="FC_1")(norm0)
    norm1 = BatchNormalization()(d1)
    d2 = Dense(32, activation='relu', name="FC_2")(norm1)
    norm2 = BatchNormalization()(d2)
    d3 = Dense(32, activation='relu', name="FC_3")(norm2)
    norm3 = BatchNormalization()(d3)
    output  = Dense(1, activation = 'sigmoid', name="predictions")(norm_d3)
    return Model(inputs=[q1_input, q2_input], outputs=output)


def args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='input_folder')
    return parser.parse_args()

if __name__ == "__main__":
    
    train_folder = args().input_folder
    
    DATA_HOME = train_folder
    TRAIN_Q1 = os.path.join(DATA_HOME, "q1_X_train_padded.npy")
    TRAIN_Q2 = os.path.join(DATA_HOME, "q2_X_train_padded.npy")
    TRAIN_Y = os.path.join(DATA_HOME, "y_train.npy")
    WORD_DIC = os.path.join(DATA_HOME, "word_index.pkl")
    
    # lstm parameters

    hidden_lstm = 512
    dropout_embedding = 0.1
    dropout_lstm = 0.1
    batch_size = 256
    epochs=10
    
    
    # make embedding matrix


    filepath = os.path.join(MODELS, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')

    print("Loading training data ....")
    q1_train, q2_train, y_train, word_index  = load_data()
    
    # make embedding matrix
    embedding_matrix = makeEM(MAX_NB_WORDS, word_index)
    logs = keras.callbacks.TensorBoard(log_dir=LOGS, 
                    histogram_freq=0, 
                    write_graph=True, 
                    write_images=False)

    save_model = keras.callbacks.ModelCheckpoint(filepath, 
                    monitor='val_loss', 
                    verbose=0, 
                    save_best_only=True, 
                    save_weights_only=False, 
                    mode='auto', 
                    period=1)

    print("Building Model ... ")
    model = shared_lstm(
                    embedding_matrix=embedding_matrix,
                    maxlen=MAXLEN,
                    vocab_size=MAX_NB_WORDS,
                    hidden_lstm=hidden_lstm,
                    batch_size=batch_size,
                    dropout_lstm=dropout_lstm,
                    dropout_embedding=dropout_embedding
                    )
     
    print("Compiling Model ....")
    model.compile(optimizer="rmsprop", 
                    loss="binary_crossentropy", 
                    metrics=["accuracy"])


    print("Fitting Model.....")
    model.fit([q1_train, q2_train],
                    y_train, 
                    epochs=epochs, 
                    batch_size=batch_size,
                    validation_split=0.1,
                    shuffle=True,
                    callbacks=[logs, save_model])

    print('Done training')
