import numpy as np
import seaborn as sns
import glob
import ast
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report


import pandas as pd
import os
from datetime import datetime
from sys import argv

from constants import *
from mem_unit import MemModule, MemoryUnit
import matplotlib.pyplot as plt

from pathlib import Path


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
    encoder = (
        MemModule(shrink_thresh=shrink_thresh, mem_dim=mem_dim, fea_dim=layer_list[-1])(
            encoder
        )
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

    return x_train_attack


def __map_memory(weight, input):
    att_weight = tf.matmul(input, weight, transpose_b=True)
    att_weight = tf.nn.softmax(att_weight)
    return att_weight


def load_encoder(filepath):
    return keras.models.load_model(filepath / "encoder")


def load_dnn(filepath):
    return keras.models.load_model(filepath / "dnn")


def encode_data(encoder, in_data, n_array=50):
    # Encode training data
    temp = []
    in_data = tf.convert_to_tensor(in_data)

    for fraction in np.array_split(in_data.numpy(), n_array):
        fraction = encoder[0](fraction)
        for a in encoder[1:-1]:
            fraction = a(fraction)

        fraction = __map_memory(encoder[-1].memory.weight, fraction)

        temp.append(fraction)

    return tf.convert_to_tensor(np.concatenate(temp))


def map_memory(
    folder,
    filenames,
    num_trials=10,
    num_epochs=5,
    layers=[60, 45],
    input_dims=78,
    mem_dim=250,
    learning_rate=0.0001,
    mem_module=None,
    shrink_thresh=0.004,
    batch_size=128,
    benign="BENIGN",
):
    autoencoder = make_auto_encoder(
        layers, input_dims, mem_dim, learning_rate, mem_module, shrink_thresh
    )
    x_train = pd.read_hdf(filenames.get("x_train"))
    y_train = pd.read_hdf(filenames.get("y_train"))
    y_train = y_train.reset_index(drop=True)

    x_val = pd.read_hdf(filenames.get("x_val"))
    y_val = pd.read_hdf(filenames.get("y_val"))
    y_val = y_val.reset_index(drop=True)

    x_test = pd.read_hdf(filenames.get("x_test"))
    y_test = pd.read_hdf(filenames.get("y_test"))
    y_test = y_test.reset_index(drop=True)

    x_attack = ret_specific_data(x_train, y_train, benign, True)
    x_val = ret_specific_data(x_val, y_val, benign, True)

    categories = [
        "BENIGN",
        "DoS Hulk",
        "PortScan",
        "DDoS",
        "DoS GoldenEye",
        "FTP-Patator",
        "SSH-Patator",
        "DoS slowloris",
        "DoS Slowhttptest",
        "Bot",
        "Web Attack Brute Force",
        "Web Attack XSS",
        "Infiltration",
        "Web Attack Sql Injection",
        "Heartbleed",
    ]

    for iter in range(0, num_trials):
        norm_data = []

        enc = autoencoder.layers[: (len(autoencoder.layers) // 2) + 1]
        for cat in categories:
            test_data = ret_specific_data(x_test, y_test, cat, negation=False)

            encoded = encode_data(enc, test_data)
            arr = tf.reduce_sum(encoded, 0).numpy()
            norm = np.linalg.norm(arr)
            normalized = arr / norm

            norm_data.append(normalized)

        # Change path to be result folder
        result_folder = folder / str(iter)

        result_folder.mkdir(parents=True)

        ax = sns.heatmap(norm_data, yticklabels=categories)
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(result_folder / "mapped_memory.png")
        fig.clear()

        count = 0
        for cat in categories:
            with open(result_folder / (cat + ".npy"), "wb") as f:
                np.save(f, norm_data[count])
            count += 1

        autoencoder.fit(
            x_attack,
            x_attack,
            epochs=num_epochs,
            batch_size=batch_size,
            validation_data=(x_val, x_val),
        )


def main():
    if len(argv) >= 2:
        print("inside")
        filenames = get_CICIDS_2017()
        for arg in argv[1:]:
            print(arg)
            map_memory(Path(arg), filenames)


if __name__ == "__main__":
    main()
