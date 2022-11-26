import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential  # Model, Sequential

# from tensorflow.keras.layers import Dense, Layer, dot, Softmax
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import pandas as pd
from os import mkdir
from datetime import datetime
from sys import argv

from constants import *
from mem_unit import MemModule, MemoryUnit

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from pathlib import Path


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

    # Generate trained encoder
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

    # Encode the test data for classifying
    in_x_train = encode_data(encoder, in_x_train)

    if in_x_val is not None:
        in_x_val = encode_data(encoder, in_x_val)

    # Generate dnn classifier
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


def eval_model(encoder, dnn, x_test, y_test, labels, dnn_batch_size=128):
    x_test = encode_data(encoder, x_test)

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


def ret_easy_nsl(filenames):
    x_train = pd.read_hdf(filenames.get("x_train"))
    y_train = pd.read_hdf(filenames.get("y_train"))

    x_test = pd.read_hdf(filenames.get("x_test_easy"))
    y_test = pd.read_hdf(filenames.get("y_test_easy"))

    return x_train, y_train, x_test, y_test


def ret_cic2017(filenames):
    x_train = pd.read_hdf(filenames.get("x_train"))
    y_train = pd.read_hdf(filenames.get("y_train"))

    x_val = pd.read_hdf(filenames.get("x_val"))
    y_val = pd.read_hdf(filenames.get("y_val"))

    x_test = pd.read_hdf(filenames.get("x_test"))
    y_test = pd.read_hdf(filenames.get("y_test"))

    return x_train, y_train, x_val, y_val, x_test, y_test


def ret_merge(filenames):
    x_merge = tf.convert_to_tensor(pd.read_hdf(filenames.get("x_merge")))
    y_merge = pd.read_hdf(filenames.get("y_merge"))

    return x_merge, y_merge


def ret_test(filenames):
    x_test = pd.read_hdf(filenames.get("x_test"))
    y_test = pd.read_hdf(filenames.get("y_test"))

    return x_test, y_test


def ret_attack(filenames):
    return tf.convert_to_tensor(pd.read_hdf(filenames.get("x_attack")))


def train_comp_2017(
    result_folder,
    auto_layers,
    dnn_layers,
    split=0.8,
    enc_epochs=150,
    enc_batch_size=512,
    enc_learning_rate=0.0001,
    dnn_epochs=150,
    dnn_batch_size=512,
    dnn_learning_rate=0.0001,
    mem_dim=1500,
    easy=True,
    attack=True,
    num_classes=15,
    max_num=None,
    shrink_thresh=0,
):
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

    filenames = get_CICIDS_2017()

    x_train, y_train, x_val, y_val, x_test, y_test = ret_cic2017(filenames)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    if max_num:
        print("Max num is ", max_num)
        rus = RandomUnderSampler({"BENIGN": max_num})
        x_train, y_train = rus.fit_resample(x_train, y_train)

    result_folder = result_folder / "encoders"
    for cat in categories:
        final_folder = result_folder / cat
        x_attack, y_attack = ret_specific_data(x_train, y_train, cat, False)
        x_attack_val, y_attack_val = ret_specific_data(x_val, y_val, cat, False)

        auto_encoder = make_auto_encoder(
            auto_layers, 78, mem_dim, enc_learning_rate, shrink_thresh=(shrink_thresh)
        )

        encoder, enc_history = train_encoder(
            x_attack,
            x_attack_val,
            auto_layers,
            auto_encoder,
            enc_epochs,
            enc_batch_size,
            split,
        )

        save_model(auto_encoder, None, final_folder)

        with open(final_folder / "encoder_history.txt", "w") as f:
            f.write(str(enc_history.history))

        x_attack = None
        x_attack_val = None


def train_NSL(
    result_folder,
    auto_layers,
    dnn_layers,
    split=0.8,
    enc_epochs=150,
    enc_batch_size=128,
    enc_learning_rate=0.0001,
    dnn_epochs=150,
    dnn_batch_size=128,
    dnn_learning_rate=0.0001,
    mem_dim=1500,
    easy=True,
    attack=True,
    num_classes=5,
    max_num=None,
):

    # Read in NSL_KDD dataset
    print("Starting")
    filenames = get_all_data()
    x_val = pd.read_hdf(filenames.get("x_val"))
    y_val = pd.read_hdf(filenames.get("y_val"))

    if easy:
        if max_num:
            print("Non zero max num")
            x_train, y_train, x_test, y_test = ret_easy_nsl(filenames)
            rus = SMOTE(
                sampling_strategy={"u2r": max_num, "r2l": max_num, "probe": max_num}
            )
            x_train, y_train = rus.fit_resample(x_train, y_train)
            pass

        else:
            x_train, y_train, x_test, y_test = ret_easy_nsl(filenames)
            print(y_train.value_counts())

    if attack:
        # Retrieve attack data sample only
        y_train = y_train.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)
        x_train_attack, y_train_attack = ret_specific_data(
            x_train, y_train, "normal", True
        )
        x_attack_val, y_attack_val = ret_specific_data(x_val, y_val, "normal", True)

    y_train, y_test, le = encode_labels(y_train, y_test)
    y_val = le.transform(y_val)
    labels = le.inverse_transform([0, 1, 2, 3, 4])

    # Generate DNN and Autoencoder
    auto_encoder = make_auto_encoder(auto_layers, 122, mem_dim, enc_learning_rate)
    dnn = make_dnn(dnn_layers, num_classes, dnn_learning_rate)

    # Train both models
    dnn, dnn_history, encoder, enc_history = val_split_training(
        x_train,
        x_train_attack,
        y_train,
        x_val,
        y_val,
        x_attack_val,
        auto_layers,
        auto_encoder,
        dnn,
        split,
        enc_epochs,
        enc_batch_size,
        dnn_epochs,
        dnn_batch_size,
    )

    class_string, class_dict, test_results = eval_model(
        encoder, dnn, x_test, y_test, labels, dnn_batch_size
    )

    save_model(auto_encoder, dnn, result_folder)

    # Write down results
    write_results(
        enc_history, dnn_history, test_results, class_string, class_dict, result_folder
    )


def train_CIC2017(
    result_folder,
    auto_layers,
    dnn_layers,
    split=0.8,
    enc_epochs=150,
    enc_batch_size=512,
    enc_learning_rate=0.0001,
    dnn_epochs=150,
    dnn_batch_size=512,
    dnn_learning_rate=0.0001,
    mem_dim=1500,
    easy=True,
    attack=True,
    num_classes=15,
    max_num=None,
    shrink_thresh=0,
):

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

    filenames = get_CICIDS_2017()

    if max_num:
        x_train, y_train, x_val, y_val, x_test, y_test = ret_cic2017(filenames)
        y_train = y_train.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)
        rus_dict = {}
        for a in categories[4:]:
            rus_dict[a] = max_num

        rus = SMOTE(sampling_strategy=rus_dict, n_jobs=-1)
        x_train, y_train = rus.fit_resample(x_train, y_train)

        print(y_train.value_counts())

    else:
        x_train, y_train, x_val, y_val, x_test, y_test = ret_cic2017(filenames)
        y_train = y_train.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)

    x_attack_val = None
    y_attack_val = None

    if attack and max_num:
        x_attack, y_attack = ret_specific_data(x_train, y_train, "BENIGN", True)
        x_attack_val, y_attack_val = ret_specific_data(x_val, y_val, "BENIGN", True)

    elif attack:
        x_attack, y_attack = ret_specific_data(x_train, y_train, "BENIGN", True)
        x_attack_val, y_attack_val = ret_specific_data(x_val, y_val, "BENIGN", True)

    else:
        pass

    y_train, y_test, le = encode_labels(y_train, y_test)
    y_val = le.transform(y_val)
    labels = le.inverse_transform([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

    auto_encoder = make_auto_encoder(
        auto_layers, 78, mem_dim, enc_learning_rate, shrink_thresh=(shrink_thresh)
    )
    dnn = make_dnn(dnn_layers, num_classes, dnn_learning_rate)

    # Train both models
    dnn, dnn_history, encoder, enc_history = val_split_training(
        x_train,
        x_attack,
        y_train,
        x_val,
        y_val,
        x_attack_val,
        auto_layers,
        auto_encoder,
        dnn,
        split,
        enc_epochs,
        enc_batch_size,
        dnn_epochs,
        dnn_batch_size,
    )

    x_test = pd.read_hdf(filenames.get("x_test"))
    class_string, class_dict, test_results = eval_model(
        encoder, dnn, x_test, y_test, labels, dnn_batch_size
    )

    # save_model(encoder, dnn, result_folder)
    save_model(auto_encoder, dnn, result_folder)

    # Write down results
    write_results(
        enc_history, dnn_history, test_results, class_string, class_dict, result_folder
    )


def ret_2018_train(filenames):
    return [
        pd.read_hdf(filenames.get("x_train"), key="my_key"),
        pd.read_hdf(filenames.get("y_train"), key="my_key"),
    ]


def ret_2018_val(filenames):
    return [
        pd.read_hdf(filenames.get("x_val"), key="my_key"),
        pd.read_hdf(filenames.get("y_val"), key="my_key"),
    ]


def ret_2018_test(filenames):
    return [
        pd.read_hdf(filenames.get("x_test"), key="my_key"),
        pd.read_hdf(filenames.get("y_test"), key="my_key"),
    ]


def train_CIC2018(
    result_folder,
    auto_layers,
    dnn_layers,
    split=0.8,
    enc_epochs=150,
    enc_batch_size=512,
    enc_learning_rate=0.0001,
    dnn_epochs=150,
    dnn_batch_size=512,
    dnn_learning_rate=0.0001,
    mem_dim=1500,
    easy=True,
    attack=True,
    num_classes=15,
    max_num=None,
    shrink_thresh=0,
):

    filenames = get_CICIDS_2018()

    start_time = time.time()

    if max_num:
        x_train, y_train = ret_2018_train(filenames)
        y_train = y_train.reset_index(drop=True)

        rus = RandomUnderSampler({"Benign": max_num})
        x_train, y_train = rus.fit_resample(x_train, y_train)
        y_train = y_train.reset_index(drop=True)

        pass

    else:
        x_train, y_train = ret_2018_train(filenames)
        y_train = y_train.reset_index(drop=True)

    print("Loaded training data ", (time.time() - start_time))
    start_time = time.time()

    x_val, y_val = ret_2018_val(filenames)
    y_val.reset_index(drop=True, inplace=True)

    print("Loaded validation data ", (time.time() - start_time))
    start_time = time.time()

    x_attack_val = None
    y_attack_val = None

    if attack and max_num:
        x_attack, y_attack = ret_specific_data(x_train, y_train, "Benign", True)
        x_attack_val, y_attack_val = ret_specific_data(x_val, y_val, "Benign", True)

    elif attack:
        x_attack, y_attack = ret_specific_data(x_train, y_train, "Benign", True)
        x_attack_val, y_attack_val = ret_specific_data(x_val, y_val, "Benign", True)

    else:
        pass

    print("Loaded attack data ", (time.time() - start_time))
    start_time = time.time()

    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_val = le.transform(y_val)

    labels = le.inverse_transform([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

    auto_encoder = make_auto_encoder(
        auto_layers, 77, mem_dim, enc_learning_rate, shrink_thresh=(shrink_thresh)
    )
    dnn = make_dnn(dnn_layers, num_classes, dnn_learning_rate)

    print("Starting training  ", (time.time() - start_time))
    start_time = time.time()

    # Train both models
    dnn, dnn_history, encoder, enc_history = val_split_training(
        x_train,
        x_attack,
        y_train,
        x_val,
        y_val,
        x_attack_val,
        auto_layers,
        auto_encoder,
        dnn,
        split,
        enc_epochs,
        enc_batch_size,
        dnn_epochs,
        dnn_batch_size,
    )

    print("Finised training  ", (time.time() - start_time))
    start_time = time.time()

    x_test, y_test = ret_2018_test(filenames)
    y_test = le.transform(y_test)

    print("Loaded test data ", (time.time() - start_time))
    start_time = time.time()
    class_string, class_dict, test_results = eval_model(
        encoder, dnn, x_test, y_test, labels, dnn_batch_size
    )

    save_model(auto_encoder, dnn, result_folder)

    # Write down results
    write_results(
        enc_history, dnn_history, test_results, class_string, class_dict, result_folder
    )


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
    return result_folder / time


def NSL(line):
    full_args = line.split(",")
    args = full_args[1:]

    args[0] = result_folder_date(args[0])
    Path.mkdir(args[0], parents=True)
    write_param(args[0], args)
    enc_layers = list((int(x) for x in args[1].split(":")))
    dnn_layers = list(int(x) for x in args[2].split(":"))
    if args[14] == "None":
        args[14] = None
    else:
        args[14] = int(args[14])

    train_NSL(
        args[0],
        enc_layers,
        dnn_layers,
        float(args[3]),
        int(args[4]),
        int(args[5]),
        float(args[6]),
        int(args[7]),
        int(args[8]),
        float(args[9]),
        int(args[10]),
        args[11],
        args[12],
        int(args[13]),
        args[14],
    )
    pass


def CICIDS2017(line):
    full_args = line.split(",")
    args = full_args[1:]

    enc_layers = list((int(x) for x in args[1].split(":")))
    dnn_layers = list(int(x) for x in args[2].split(":"))

    if args[14] == "None" or args[14] == "None\n":
        args[14] = None
    else:
        args[14] = int(args[14])

    for num in range(int(args[16])):
        result_folder = result_folder_date(args[0])
        Path.mkdir(result_folder, parents=True)
        write_param(result_folder, args)
        train_CIC2017(
            result_folder=result_folder,
            auto_layers=enc_layers,
            dnn_layers=dnn_layers,
            split=float(args[3]),
            enc_epochs=int(args[4]),
            enc_batch_size=int(args[5]),
            enc_learning_rate=float(args[6]),
            dnn_epochs=int(args[7]),
            dnn_batch_size=int(args[8]),
            dnn_learning_rate=float(args[9]),
            mem_dim=int(args[10]),
            easy=args[11],
            attack=args[12],
            num_classes=int(args[13]),
            max_num=args[14],
            shrink_thresh=float(args[15]),
        )
    pass


def COMP_CICIDS2017(line):
    full_args = line.split(",")
    args = full_args[1:]

    args[0] = result_folder_date(args[0])

    Path.mkdir(args[0], parents=True)
    write_param(args[0], args)
    enc_layers = list((int(x) for x in args[1].split(":")))
    dnn_layers = list(int(x) for x in args[2].split(":"))
    if args[14] == "None" or args[14] == "None\n":
        args[14] = None
    else:
        args[14] = int(args[14])

    train_comp_2017(
        args[0],
        enc_layers,
        dnn_layers,
        float(args[3]),
        int(args[4]),
        int(args[5]),
        float(args[6]),
        int(args[7]),
        int(args[8]),
        float(args[9]),
        int(args[10]),
        args[11],
        args[12],
        int(args[13]),
        args[14],
        float(args[15]),
    )


def CICIDS2018(line):
    full_args = line.split(",")
    args = full_args[1:]

    enc_layers = list((int(x) for x in args[1].split(":")))
    dnn_layers = list(int(x) for x in args[2].split(":"))

    if args[14] == "None" or args[14] == "None\n":
        args[14] = None
    else:
        args[14] = int(args[14])

    for num in range(int(args[16])):
        result_folder = result_folder_date(args[0])
        Path.mkdir(result_folder, parents=True)
        write_param(result_folder, args)
        train_CIC2018(
            result_folder=result_folder,
            auto_layers=enc_layers,
            dnn_layers=dnn_layers,
            split=float(args[3]),
            enc_epochs=int(args[4]),
            enc_batch_size=int(args[5]),
            enc_learning_rate=float(args[6]),
            dnn_epochs=int(args[7]),
            dnn_batch_size=int(args[8]),
            dnn_learning_rate=float(args[9]),
            mem_dim=int(args[10]),
            easy=args[11],
            attack=args[12],
            num_classes=int(args[13]),
            max_num=args[14],
            shrink_thresh=float(args[15]),
        )
    pass


def command_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("1"):
                NSL(line)

            elif line.startswith("2"):
                NSL_COMP(line)

            elif line.startswith("3"):
                CICIDS2017(line)

            elif line.startswith("5"):
                CICIDS2018(line)

            elif line.startswith("7"):
                COMP_CICIDS2017(line)
    pass


def main():
    if len(argv) == 2:
        cleaned_path = str(argv[1].replace("\\", "/"))
        filename = Path(cleaned_path)
        command_file(filename)
        print("Finished")

    else:
        print("Give input and output")
    pass


if __name__ == "__main__":
    main()
