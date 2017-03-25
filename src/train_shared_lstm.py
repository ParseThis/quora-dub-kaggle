import os.path
import sys
import pdb
import json
import pickle
from datetime import datetime


import numpy as np
from keras import backend as K
from keras.models import Model
import keras.layers
from keras.layers import LSTM, Merge, merge, Embedding, Dropout, Input, Dense

import argparse

HOME = os.path.join(os.path.expanduser('~'), "quora-duplicate-questions")
MODELS = os.path.join(HOME, 'models')
LOGS = os.path.join(HOME, 'logs')

class ShowPredictions(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        print(self.model.predict(batch, batch_size=len(batch)))


def load_data():

    q1_X_train = np.load(TRAIN_Q1)           
    q2_X_train = np.load(TRAIN_Q2)           
    y_train = np.load(TRAIN_Y)
    # word_index = pickle.load(open(WORD_DIC, 'rb'))

    return q1_X_train, q2_X_train, y_train

def shared_lstm(maxlen=None,
                vocab_size=None,
                hidden_lstm=None,
                batch_size=None,
                dropout_lstm=None,
                dropout_embedding=None):
                
    q1_input = Input(shape=(maxlen,), name='q1_input')
    q2_input = Input(shape=(maxlen,), name='q2_input')
    
    q1_embed = Embedding(vocab_size+2, 
                        hidden_lstm, 
                        input_length=maxlen,
                        mask_zero=True,
                        name="q1_embedding")(q1_input)
    q2_embed = Embedding(vocab_size + 2,
                        hidden_lstm,
                        input_length=maxlen,
                        mask_zero=True,
                        name="q2_embedding")(q2_input)

    dropout_q1 = Dropout(dropout_embedding, name="dropout_q1")(q1_embed)
    dropout_q2 = Dropout(dropout_embedding, name="dropout_q2")(q2_embed)

    lstm = LSTM(64, name="shared_lstm")
    q1_shared_lstm = lstm(dropout_q1)
    q2_shared_lstm = lstm(dropout_q2)

    gathered = keras.layers.concatenate([q1_shared_lstm, q2_shared_lstm], axis=-1)

    dropout_gathered = Dropout(dropout_lstm, name="gathered_dropout")(gathered)
    d1 = Dense(32, activation='relu', name="FC_1")(dropout_gathered)
    d2 = Dense(32, activation='relu', name="FC_2")(d1)
    d3 = Dense(32, activation='relu', name="FC_3")(d2)
    dropout_d3 = Dropout(dropout_lstm, name="dropout_after_FC")(d3)

    output  = Dense(1, activation = 'sigmoid', name="predictions")(dropout_d3)
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


    maxlen = 50
    vocab_size = 50000

    # lstm parameters

    hidden_lstm = 128
    dropout_embedding = 0.8
    dropout_lstm = 0.8
    batch_size = 16
    epochs=5

    print("Loading training data ....")
    q1_train, q2_train, y_train  = load_data()

    clbks = keras.callbacks.TensorBoard(log_dir=LOGS, 
                    histogram_freq=0, 
                    write_graph=True, 
                    write_images=False)

    print("Building Model ... ")
    model = shared_lstm(
                    maxlen=maxlen,
                    vocab_size=vocab_size,
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
                    callbacks=[clbks])

    # save weights
    day = datetime.now().strftime("%A_%d_%m_%_H_%-M")
    
    try: 
        folder = os.path.join(MODELS, day)
        os.makedirs(folder)
    except FileExistError:
        pass

    print('Storing model weights in: ', folder)
    model.save_weights(os.path.join(folder, 'shared_model_weights.hdf5'))

    # save model
    print('Storing model in: ', folder)
    f = open(os.path.join(MODELS,  'shared_model.json'), 'w').write(model.to_json())
    print('Done training')
