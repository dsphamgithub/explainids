import numpy as np

import ast


import pandas as pd
import os

from sys import argv

from constants import *

from shutil import copyfile


from pathlib import Path


class result_block:
    def __init__(self):
        self.params = {}
        self.results = []
        self.avg_result = {}

    def __str__(self):
        return str(self.params) + str(self.results)


def read_in_results(argv):
    results = []

    for arg in argv:
        filepath = Path(arg)
        for path in filepath.iterdir():
            params_path = path / "params.txt"
            classification = path / "class_dict.txt"
            params = ""

            if params_path.exists() and classification.exists():
                with open(params_path, "r") as f:
                    lines = f.readlines()
                    if lines[-1] == "\n":
                        lines = lines[:-1]

                    res_dict = {
                        lines[i][:-1]
                        .split("=")[0]
                        .strip(" "): lines[i][:-1]
                        .split("=")[1]
                        .strip(" ")
                        for i in range(0, len(lines))
                    }

                res_block = None
                for res in results:
                    if res.params == res_dict:
                        res_block = res

                if res_block is None:
                    res_block = result_block()
                    res_block.params = res_dict
                    results.append(res_block)

                with open(classification, "r") as f:

                    perf_dict = ast.literal_eval(f.readline())

                res_block.results.append(perf_dict)

    return results


def average_results(results):
    avg_results = []
    for res in results:

        count = 0
        attack_cat = {}
        for dict in res.results:

            count += 1
            for key in dict:
                if key != "accuracy":
                    try:
                        res.avg_result[key]
                    except:
                        res.avg_result[key] = {
                            "precision": 0,
                            "recall": 0,
                            "f1-score": 0,
                            "support": 0,
                        }

                    for metric in dict[key]:
                        res.avg_result[key][metric] += dict[key][metric]

                else:
                    try:
                        res.avg_result[key] += dict[key]
                    except:
                        res.avg_result[key] = dict[key]

        for key in res.avg_result:
            if key != "accuracy":
                for metric in res.avg_result[key]:
                    res.avg_result[key][metric] /= count
            else:
                res.avg_result[key] /= count

    return results


def sort_results(results):
    mac_count = 0
    wei_count = 0

    for res in results:
        if res.avg_result["macro avg"]["f1-score"] > 0.65:
            print(res.avg_result["macro avg"]["f1-score"])
            mac_count += 1
            print(res.params)

        if res.avg_result["weighted avg"]["f1-score"] > 0.979:
            res.avg_result["weighted avg"]["f1-score"]
            print(res.avg_result)
            print(res.params)
            wei_count += 1

    print(mac_count)
    print(wei_count)
    pass


def write_results(path, results, params):

    if not path.exists():
        os.mkdir(path)

    count = 0

    with open((path / "total.csv"), "w") as all_file:
        all_file.write("params,macro_avg_f1,weighted_avg_f1\n")
        for res in results:
            filename = ""
            for a in params:
                print(res.params)
                filename += str(res.params[a]) + ","

            all_file.write(filename.replace(",", " ") + ",")
            filename += ".csv"
            with open((path / filename), "w") as f:
                for a in params:
                    f.write(str(a) + "," + str(res.params[a]) + ",")
                f.write("\n")

                # Extract just the categories
                temp = list(res.avg_result.keys())
                print(temp)
                # return
                f.write("category,precision,recall,f1-score\n")
                for key in temp:
                    if key != "accuracy":
                        print(key)
                        f.write(str(key) + ",")
                        for a in list(res.avg_result[key])[:-1]:
                            f.write(str(res.avg_result[key][a]) + ",")
                        f.write("\n")

                f.write("accuracy,")
                f.write(str(res.avg_result["accuracy"]) + "\n")

                # Write out to all file
                for key in temp[-2:]:
                    all_file.write(str(res.avg_result[key]["f1-score"]) + ",")
                all_file.write("\n")
            count += 1
    pass


def main():
    if len(argv) == 2:
        results = read_in_results(argv[1:])
        results = average_results(results)
        print(results)

        write_results(
            argv[-1] / Path("processed_results"),
            results,
            ["mem_dim", "shrink_thresh", "max_num"],
        )


if __name__ == "__main__":
    main()
