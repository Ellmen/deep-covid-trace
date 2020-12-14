from random import Random

from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat
from Bio import SeqIO

from tensorflow.keras.layers import (
    Input,
    Conv1D,
    Dense,
    Activation,
    Flatten,
    Dropout,
    GlobalMaxPooling1D,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


def test_model(x_test, y_test):
    saved_model = load_model('./models/best_cnn_model_100_v5.h5')
    test_loss, test_acc = saved_model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy: {}'.format(test_acc))


def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('CNN accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


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


def convert_data(fasta_path='./data/balanced_variable_seqs.fasta'):
    regions = []
    seqs = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        sep_idx = record.description.rfind('|')
        region = record.description[sep_idx+1:]
        regions.append(region)
        seqs.append(str(record.seq))

    region_options = [r for r in set(regions)]
    x = np.array([seq_to_pwm(seq) for seq in seqs]).astype(np.float32)
    y = to_categorical([region_options.index(r) for r in regions])
    savemat('{}.mat'.format(fasta_path[:fasta_path.rfind('.')]), {'X': x, 'Y': y})



def train(x, y):
    # TODO: Optimize models, parameters
    input_shape = (None, 4)
    dropout_pool = 0.5
    dropout_dense = 0.5

    es = EarlyStopping(monitor='val_loss', mode='max', verbose = 1, patience=300)
    mc = ModelCheckpoint('./models/best_cnn_model_100_v5.h5', monitor='val_accuracy', mode='max', verbose = 1, save_best_only=True)

    batch_size = 10
    epochs = 200
    num_filters = 200
    filter_len = 10

    model = Sequential([
        Conv1D(filters=num_filters, kernel_size=filter_len, activation='relu', input_shape=input_shape),
        GlobalMaxPooling1D(),
        # Dropout(dropout_pool),
        Dense(num_filters, activation='relu'),
        # Dropout(dropout_dense),
        Dense(y.shape[1], activation='softmax')
    ])
    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    history = model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks = [es, mc])
    plot_history(history)


if __name__ == '__main__':
    # convert_data()
    data = loadmat('./data/balanced_variable_seqs.mat')
    # data = loadmat('test.mat')
    seed = 1
    X = data['X']
    Y = data['Y']
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(Y)
    splt = int(len(Y) * 0.9)
    x_train = X[:splt]
    y_train = Y[:splt]
    x_test = X[splt:]
    y_test = Y[splt:]
    del data, X, Y
    # train(x_train, y_train)
    test_model(x_test, y_test)
