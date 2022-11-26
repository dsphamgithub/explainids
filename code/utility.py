import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential  # Model, Sequential

# from tensorflow.keras.layers import Dense, Layer, dot, Softmax
from sklearn.preprocessing import LabelEncoder

# from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

from pathlib import Path
from os import mkdir
from datetime import datetime
from constants import *
from sys import argv
from mem_unit import *


def load_autoencoder(filepath):
    if not (filepath / "encoder").exists():
        raise FileNotFoundError
    enc = keras.models.load_model(filepath / "encoder")
    return enc


def load_encoder(filepath):
    if not (filepath / "encoder").exists():
        raise FileNotFoundError
    enc = keras.models.load_model(filepath / "encoder")
    enc = enc.layers[: (len(enc.layers) // 2) + 1]
    return enc


def load_dnn(filepath):
    return keras.models.load_model(filepath / "dnn")


def ret_specific_data(x_train, y_train, filter, negation):
    x_train_attack = x_train

    x_train_attack["attack_type"] = y_train

    # If we don't want it to match
    if negation:
        x_train_attack = x_train_attack.loc[x_train_attack["attack_type"] != filter]

    else:
        x_train_attack = x_train_attack.loc[x_train_attack["attack_type"] == filter]
    # print(x_train_attack)
    y_train_attack = x_train_attack.pop("attack_type")
    x_train.pop("attack_type")

    return x_train_attack


def write_param(result_folder, line):
    with open(result_folder / "params.txt", "w") as f:
        f.write("auto_layers = " + str(line[1]) + "\n")
        f.write("dnn_layer = " + str(line[2]) + "\n")
        f.write("split = " + str(line[3]) + "\n")
        f.write("enc_epoch = " + str(line[4]) + "\n")
        f.write("enc_batch_size = " + str(line[5]) + "\n")
        f.write("enc_learning_rate = " + str(line[6]) + "\n")
        f.write("dnn_epochs = " + str(line[7]) + "\n")
        f.write("dnn_batch_size = " + str(line[8]) + "\n")
        f.write("dnn_learning_rate = " + str(line[9]) + "\n")
        f.write("mem_dim = " + str(line[10]) + "\n")
        f.write("easy = " + str(line[11]) + "\n")
        f.write("attack = " + str(line[12]) + "\n")
        f.write("num_classes = " + str(line[13]) + "\n")
        f.write("max_num = " + str(line[14]) + "\n")
        f.write("shrink_thresh = " + str(line[15]) + "\n")


def result_folder_date(result_folder):
    result_folder = Path(result_folder)

    time = str(datetime.now()).replace(":", ",")
    print(result_folder / time)
    return result_folder / time


def ret_cic2017(filenames):
    x_train = pd.read_hdf(filenames.get("x_train"))
    y_train = pd.read_hdf(filenames.get("y_train"))

    x_val = pd.read_hdf(filenames.get("x_val"))
    y_val = pd.read_hdf(filenames.get("y_val"))

    x_test = pd.read_hdf(filenames.get("x_test"))
    y_test = pd.read_hdf(filenames.get("y_test"))

    return x_train, y_train, x_val, y_val, x_test, y_test


def write_results(
    enc_history, dnn_history, test_results, classification, class_dict, folder_name
):

    with open(folder_name / "encoder_history.txt", "w") as f:
        f.write(str(enc_history.history))

    with open(folder_name / "dnn_history.txt", "w") as f:
        f.write(str(dnn_history.history))

    with open(folder_name / "test_results.txt", "w") as f:
        f.write(str(test_results))

    with open(folder_name / "classification.txt", "w") as f:
        f.write(str(classification))

    with open(folder_name / "class_dict.txt", "w") as f:
        f.write(str(class_dict))


def save_model(encoder, dnn, result_folder):
    if encoder is not None:
        encoder.save(result_folder / "encoder")

    if dnn is not None:
        dnn.save(result_folder / "dnn")


def make_auto_encoder(
    layer_list,
    input_dims,
    mem_dim=1500,
    learning_rate=0.0001,
    mem_module=None,
    shrink_thresh=0,
):
    inputs = keras.Input(shape=(input_dims,))

    # Making encoder
    encoder = layers.Dense(layer_list[0], activation="relu")(inputs)
    for a in layer_list[1:]:
        encoder = layers.Dense(a, activation="relu")(encoder)
    if mem_dim > 0:
        encoder = (
            MemModule(
                shrink_thresh=shrink_thresh, mem_dim=mem_dim, fea_dim=layer_list[-1]
            )(encoder)
            if mem_module is None
            else mem_module(encoder)
        )

    # Making decoder
    decoder = layers.Dense(layer_list[-1], activation="relu")(encoder)
    for a in reversed(layer_list[:-1]):
        decoder = layers.Dense(a, activation="relu")(decoder)
    decoder = layers.Dense(input_dims)(decoder)

    model = keras.Model(inputs=inputs, outputs=decoder, name="AutoEncoder")
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
    )

    return model


def make_dnn(layer_list, num_categories, learning_rate=0.0001):
    dnn = Sequential()
    for b in layer_list:
        dnn.add(layers.Dense(int(b), activation="relu"))
    dnn.add(layers.Dense(num_categories, activation="softmax"))

    dnn.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )

    return dnn


def encode_data(encoder, in_data, n_array=50):

    temp = []

    in_data = tf.convert_to_tensor(in_data)

    for fraction in np.array_split(in_data.numpy(), n_array):
        fraction = encoder[0](fraction)
        for a in encoder[1:]:
            fraction = a(fraction)

        temp.append(fraction)

    return tf.convert_to_tensor(np.concatenate(temp))


def train_encoder(
    in_x_attack,
    in_x_val,
    layer_list,
    auto_encoder,
    epochs=150,
    batch_size=128,
    split=0.8,
):
    if in_x_val is not None:

        history = auto_encoder.fit(
            in_x_attack,
            in_x_attack,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(in_x_val, in_x_val),
        )

    else:
        history = auto_encoder.fit(
            in_x_attack,
            in_x_attack,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.8,
        )

    # If encoder has memory module
    if len(auto_encoder.layers) % 2 == 1:
        encoder = auto_encoder.layers[: (len(layer_list) + 2)]
    else:
        encoder = auto_encoder.layers[: (len(layer_list) + 1)]

    return [encoder, history]


# @profile
def train_dnn(
    dnn,
    enc_x_train,
    enc_x_val,
    in_y_train,
    in_y_val,
    epochs=150,
    batch_size=128,
    split=0.8,
):
    if enc_x_val is not None:
        history = dnn.fit(
            enc_x_train,
            in_y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(enc_x_val, in_y_val),
        )

    else:
        history = dnn.fit(
            enc_x_train,
            in_y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=split,
        )

    print(dnn.summary())

    return [dnn, history]


def eval_model(encoder, dnn, x_test, y_test, labels, dnn_batch_size=128):
    x_test = encode_data(encoder, x_test)

    print("Before gen_classification", x_test)

    class_string = gen_classification(dnn, x_test, dnn_batch_size, y_test, labels)
    class_dict = gen_classification(dnn, x_test, dnn_batch_size, y_test, labels, True)

    test_results = dnn.evaluate(x_test, (y_test))

    return class_string, class_dict, test_results


def ret_specific_data(x_train, y_train, filter, negation):
    x_train_attack = x_train

    x_train_attack["attack_type"] = y_train

    # If we don't want it to match
    if negation:
        x_train_attack = x_train_attack.loc[x_train_attack["attack_type"] != filter]

    else:
        x_train_attack = x_train_attack.loc[x_train_attack["attack_type"] == filter]
    y_train_attack = x_train_attack.pop("attack_type")
    x_train.pop("attack_type")

    return x_train_attack, y_train_attack


def encode_labels(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    return y_train, y_test, le


def train_model(
    in_x_train,
    in_y_train,
    in_x_attack,
    in_x_val,
    in_y_val,
    x_attack_val,
    auto_layers,
    auto_encoder,
    dnn,
    enc_epochs=150,
    enc_batch_size=128,
    dnn_epochs=150,
    dnn_batch_size=128,
    split=0.8,
):

    encoder, enc_history = train_encoder(
        in_x_attack,
        x_attack_val,
        auto_layers,
        auto_encoder,
        enc_epochs,
        enc_batch_size,
        split,
    )

    del in_x_attack
    del x_attack_val

    in_x_train = encode_data(encoder, in_x_train)

    if in_x_val is not None:
        in_x_val = encode_data(encoder, in_x_val)

    print("After encoding")
    dnn, dnn_history = train_dnn(
        dnn,
        in_x_train,
        in_x_val,
        in_y_train,
        in_y_val,
        dnn_epochs,
        dnn_batch_size,
        split=split,
    )

    return (dnn, dnn_history, encoder, enc_history)


def gen_classification(dnn, x_test, batch_size, y_test, labels, output_dict=False):
    y_pred = np.argmax(dnn.predict(x_test, batch_size), axis=1)
    return classification_report(
        y_test, y_pred, target_names=labels, output_dict=output_dict
    )


def val_split_training(
    x_train,
    x_train_attack,
    y_train,
    x_val,
    y_val,
    x_attack_val,
    auto_layers,
    auto_encoder,
    dnn,
    split=0.8,
    enc_epochs=150,
    enc_batch_size=128,
    dnn_epochs=150,
    dnn_batch_size=128,
):

    # Run the do results and write command
    dnn, dnn_history, encoder, enc_history = train_model(
        x_train,
        y_train,
        x_train_attack,
        x_val,
        y_val,
        x_attack_val,
        auto_layers,
        auto_encoder,
        dnn,
        enc_epochs,
        enc_batch_size,
        dnn_epochs,
        dnn_batch_size,
        split,
    )

    return dnn, dnn_history, encoder, enc_history
