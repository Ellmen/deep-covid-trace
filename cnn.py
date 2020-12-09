import numpy as np
from scipy.io import loadmat, savemat
from Bio import SeqIO
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    Dense,
    Activation,
    Flatten,
    Dropout,
    GlobalMaxPooling1D,
)
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


# One hot encoding of base pairs
def encode_base(base):
    bases = ['A', 'C', 'G', 'T']
    if base in bases:
        encoded = [0] * 4
        encoded[bases.index(base)] = 1
        return encoded

    return [1/4] * 4


# The position weight matrix for a given sequence
def seq_to_pwm(seq):
    return [encode_base(base) for base in seq]


def convert_data(fasta_path='./data/variable_seqs.fasta'):
    regions = []
    seqs = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        sep_idx = record.id.rfind('|')
        region = record.id[sep_idx+1:]
        regions.append(region)
        seqs.append(str(record.seq))

    region_options = [r for r in set(regions)]
    x = np.array([seq_to_pwm(seq) for seq in seqs]).astype(np.float32)
    y = to_categorical([region_options.index(r) for r in regions])
    savemat('{}.mat'.format(fasta_path[:fasta_path.rfind('.')]), {'X': x, 'Y': y})



def train(x, y):
    # TODO: Optimize models, parameters
    input_shape = (None, 4)
    dropout_pool = 0.1
    dropout_dense = 0.1

    es = EarlyStopping(monitor='val_loss', mode='max', verbose = 1, patience=300)
    mc = ModelCheckpoint('./models/best_cnn_model.h5', monitor='val_accuracy', mode='max', verbose = 1, save_best_only=True)

    batch_size = 10
    epochs = 300
    num_filters = 10
    # num_filters = 1
    filter_len = 10

    model = Sequential([
        Conv1D(filters=num_filters, kernel_size=filter_len, activation='relu', input_shape=input_shape),
        GlobalMaxPooling1D(),
        # Dropout(dropout_pool),
        Dense(num_filters, activation='relu'),
        # Dense(1, activation='relu'),
        # Dropout(dropout_dense),
        Dense(y.shape[1], activation='sigmoid')
    ])
    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks = [es, mc])


if __name__ == '__main__':
    # convert_data()
    data = loadmat('./data/variable_seqs.mat')
    # data = loadmat('test.mat')
    X = data['X']
    Y = data['Y']
    splt = int(len(Y) * 0.9)
    x_train = X[:splt]
    y_train = Y[:splt]
    x_test = X[splt:]
    y_test = Y[splt:]
    del data, X, Y
    train(x_train, y_train)
