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
from sklearn.decomposition import PCA


import pandas as pd
import os
from datetime import datetime
from sys import argv

from constants import *
from mem_unit import MemModule, MemoryUnit
import matplotlib.pyplot as plt

from pathlib import Path


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
    if not (filepath / "encoder").exists():
        raise FileNotFoundError(
            "This is reaching this point with filename " + str(filepath)
        )
    enc = keras.models.load_model(filepath / "encoder")
    enc = enc.layers[: (len(enc.layers) // 2) + 1]
    return enc


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

        # print(encoder[-1].memory)
        fraction = __map_memory(encoder[-1].memory.weight, fraction)

        temp.append(fraction)

    return tf.convert_to_tensor(np.concatenate(temp))


def plot_memory_2017(folder, filenames, enc):
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

    plot_memory(folder, filenames, enc, categories)


def plot_memory_2018(folder, filenames, enc):
    categories = [
        "Benign",
        "DDOS attack-HOIC",
        "DDoS attacks-LOIC-HTTP",
        "DoS attacks-Hulk",
        "Bot",
        "FTP-BruteForce",
        "SSH-Bruteforce",
        "Infilteration",
        "DoS attacks-SlowHTTPTest",
        "DoS attacks-GoldenEye",
        "DoS attacks-Slowloris",
        "DDOS attack-LOIC-UDP",
        "Brute Force -Web",
        "Brute Force -XSS",
        "SQL Injection",
    ]

    plot_memory(folder, filenames, enc, categories)


def plot_memory_nsl(folder, filenames, enc):
    categories = ["dos", "normal", "probe", "r2l", "u2r"]

    plot_memory(folder, filenames, enc, categories)


def plot_memory(folder, filenames, enc, categories):
    if (folder / "plot_memory").exists():
        return
        pass

    result_folder = folder / "plot_memory"
    if not (folder / "plot_memory").exists():
        os.mkdir(result_folder)

    mem = enc[-1]
    mem_contents = (mem.memory.weight).numpy()

    ax = sns.heatmap(mem_contents)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(result_folder / "plot_memory.png")
    fig.clear()

    with open(result_folder / "plot_memory.npy", "wb") as f:
        np.save(f, mem_contents)


def measure_spread_nsl(folder, filenames, enc):
    categories = ["dos", "normal", "probe", "r2l", "u2r"]

    measure_spread(folder, filenames, enc, categories)


def measure_spread_2017(folder, filenames, enc):
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

    measure_spread(folder, filenames, enc, categories)


def measure_spread_2018(folder, filenames, enc):
    categories = [
        "Benign",
        "DDOS attack-HOIC",
        "DDoS attacks-LOIC-HTTP",
        "DoS attacks-Hulk",
        "Bot",
        "FTP-BruteForce",
        "SSH-Bruteforce",
        "Infilteration",
        "DoS attacks-SlowHTTPTest",
        "DoS attacks-GoldenEye",
        "DoS attacks-Slowloris",
        "DDOS attack-LOIC-UDP",
        "Brute Force -Web",
        "Brute Force -XSS",
        "SQL Injection",
    ]

    measure_spread(folder, filenames, enc, categories)


def measure_spread(folder, filenames, enc, categories):
    map_folder = folder / "mapped_memory"
    mem_folder = folder / "plot_memory"
    input_folder = folder / "top_n"
    result_folder = folder / "spread_measure"

    if not result_folder.exists():
        os.mkdir(result_folder)
    else:
        return

    string = "Category,Threshold,Max,Count\n"
    for cat in categories:
        filename = cat + ".npy"
        mem = np.sort(np.load(map_folder / filename))[::-1]
        max = mem[0]
        count = 0
        divides = 0
        for a in mem:
            if a < (max / 2):
                max = a
                divides += 1
            count += 1

        sum = 0
        threshold = 0.7
        threshold_mem = 0.7 * np.sum(mem)
        offset = 5
        count = 0
        string += str(cat) + "," + str(threshold) + ","
        for a in mem:
            sum += a
            if sum > threshold_mem:
                print("Max ", mem[0], "Count ", count)
                string += str(mem[0]) + "," + str(count) + "\n"
                break
            count += 1
        print()
    with open(result_folder / "spread_measure.csv", "w") as f:
        f.write(string)


def best_n_nsl(folder, filenames, enc, n):
    categories = ["dos", "normal", "probe", "r2l", "u2r"]

    best_n(folder, filenames, enc, categories, n)


def best_n_2017(folder, filenames, enc, n):
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

    best_n(folder, filenames, enc, categories, n)


def best_n_2018(folder, filenames, enc, n):
    categories = [
        "Benign",
        "DDOS attack-HOIC",
        "DDoS attacks-LOIC-HTTP",
        "DoS attacks-Hulk",
        "Bot",
        "FTP-BruteForce",
        "SSH-Bruteforce",
        "Infilteration",
        "DoS attacks-SlowHTTPTest",
        "DoS attacks-GoldenEye",
        "DoS attacks-Slowloris",
        "DDOS attack-LOIC-UDP",
        "Brute Force -Web",
        "Brute Force -XSS",
        "SQL Injection",
    ]

    best_n(folder, filenames, enc, categories, n)
    pass


def best_n(folder, filenames, enc, categories, n):
    result_folder = folder / "top_n"

    if result_folder.exists():
        return
        pass
    else:
        os.mkdir(result_folder)

    mem_filepath = folder / "mapped_memory"
    plot_filepath = folder / "plot_memory"
    mem = np.load(plot_filepath / ("plot_memory.npy"))

    # 2D
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(mem)
    principalDf = pd.DataFrame(
        data=principalComponents,
        columns=["principal component 1", "principal component 2"],
    )

    pc1_max = principalDf["principal component 1"].max()
    pc1_min = principalDf["principal component 1"].min()

    pc2_max = principalDf["principal component 2"].max()
    pc2_min = principalDf["principal component 2"].min()

    ax = sns.scatterplot(
        x="principal component 1", y="principal component 2", data=principalDf
    )
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(result_folder / "principal.png")
    fig.clear()

    # 3D
    pca3D = PCA(n_components=3)
    principalComponents3D = pca3D.fit_transform(mem)
    principalDf3D = pd.DataFrame(
        data=principalComponents3D,
        columns=[
            "principal component 1",
            "principal component 2",
            "principal component 3",
        ],
    )

    pc1_3D_max = principalDf3D["principal component 1"].max()
    pc1_3D_min = principalDf3D["principal component 1"].min()

    pc2_3D_max = principalDf3D["principal component 2"].max()
    pc2_3D_min = principalDf3D["principal component 2"].min()

    pc3_3D_max = principalDf3D["principal component 3"].max()
    pc3_3D_min = principalDf3D["principal component 3"].min()

    for cat in categories:
        encoded_arr = np.load(mem_filepath / (cat + ".npy"))

        all_n = np.argsort(encoded_arr)[::-1]
        top_n = all_n[:n]

        with open(result_folder / (cat + ".npy"), "wb") as f:
            np.save(f, all_n)

        ax = sns.scatterplot(
            x="principal component 1",
            y="principal component 2",
            data=principalDf.iloc[top_n],
        )
        ax.set(ylim=(pc2_min - 0.5, pc2_max + 0.5))
        ax.set(xlim=(pc1_min - 0.5, pc1_max + 0.5))
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(result_folder / (cat + "_" + str(n) + "_" + "principal.png"))
        fig.clear()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        x = principalDf3D.iloc[top_n]["principal component 1"]
        y = principalDf3D.iloc[top_n]["principal component 2"]
        z = principalDf3D.iloc[top_n]["principal component 3"]

        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")
        ax.scatter(x, y, z)
        fig.savefig(result_folder / (cat + "_" + str(n) + "_" + "principal3D.png"))
        fig.clear()

        ax = sns.heatmap(mem[top_n], yticklabels=top_n, vmin=-1.0, vmax=1.0)
        plt.yticks(rotation=0)
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(result_folder / (cat + "_" + str(n) + ".png"))
        fig.clear()

    pass


def map_memory_2017(folder, filenames, enc):
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

    map_memory(folder, filenames, enc, categories)


def map_memory_2018(folder, filenames, enc):
    categories = [
        "Benign",
        "DDOS attack-HOIC",
        "DDoS attacks-LOIC-HTTP",
        "DoS attacks-Hulk",
        "Bot",
        "FTP-BruteForce",
        "SSH-Bruteforce",
        "Infilteration",
        "DoS attacks-SlowHTTPTest",
        "DoS attacks-GoldenEye",
        "DoS attacks-Slowloris",
        "DDOS attack-LOIC-UDP",
        "Brute Force -Web",
        "Brute Force -XSS",
        "SQL Injection",
    ]

    map_memory(folder, filenames, enc, categories)


def map_memory_nsl(folder, filenames, enc):
    categories = ["dos", "normal", "probe", "r2l", "u2r"]

    map_memory(folder, filenames, enc, categories)


def map_memory(folder, filenames, enc, categories):
    if (folder / "mapped_memory").exists():
        # return
        pass

    x_test = pd.read_hdf(filenames.get("x_test"))
    y_test = pd.read_hdf(filenames.get("y_test"))
    y_test = y_test.reset_index(drop=True)

    norm_data = []
    for cat in categories:
        test_data = ret_specific_data(x_test, y_test, cat, negation=False)

        encoded = encode_data(enc, test_data)
        arr = tf.reduce_sum(encoded, 0).numpy()

        norm = np.linalg.norm(arr)
        normalized = arr / norm

        norm_data.append(normalized)

    # Change path to be result folder
    result_folder = folder / "mapped_memory"

    if not os.path.exists(result_folder):
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

    norm_data = np.sort(norm_data)

    for max in [1.0, 0.8, 0.6, 0.4, 0.2]:
        ax = sns.heatmap(norm_data, yticklabels=categories, vmax=max)
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(result_folder / ("sorted_mapped_memory_" + str(max) + ".png"))
        fig.clear()


def main():
    n = 15

    if len(argv) >= 3:
        if argv[1] == "2017":
            print("inside cicids_2017")
            filenames = get_CICIDS_2017()
            for arg in argv[2:]:
                print(arg)
                arg = arg.strip("'").strip('"')
                for path in [x for x in Path(arg).iterdir() if x.is_dir()]:
                    print(path)
                    try:
                        enc = load_encoder(path)
                        map_memory_2017(path, filenames, enc)
                        plot_memory_2017(path, filenames, enc)
                        best_n_2017(path, filenames, enc, n)
                        measure_spread_2017(path, filenames, enc)
                    except FileNotFoundError as e:
                        print(e)
                        print("Error no encoder")

        elif argv[1] == "nsl":
            print("inside nsl")
            filenames = get_all_data()
            for arg in argv[2:]:
                print(arg)
                for path in [x for x in Path(arg).iterdir() if x.is_dir()]:
                    print(path)
                    try:
                        enc = load_encoder(path)
                        map_memory_nsl(path, filenames, enc)
                        plot_memory_nsl(path, filenames, enc)
                        best_n_nsl(path, filenames, enc, n)
                        measure_spread_nsl(path, filenames, enc)
                    except FileNotFoundError as e:
                        print(e)
                        print("Error no encoder")

        elif argv[1] == "2018":
            print("inside 2018")
            filenames = get_CICIDS_2018()
            for arg in argv[2:]:
                print(arg)
                for path in [x for x in Path(arg).iterdir() if x.is_dir()]:
                    print(path)
                    try:
                        enc = load_encoder(path)
                        map_memory_2018(path, filenames, enc)
                        plot_memory_2018(path, filenames, enc)
                        best_n_2018(path, filenames, enc, n)
                        measure_spread_2018(path, filenames, enc)
                    except FileNotFoundError as e:
                        print(e)
                        print("Error no encoder")

    else:
        print("Please enter nsl|2017|2018 and then folders you want processed")


if __name__ == "__main__":
    main()
