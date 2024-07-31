import os
import json
from collections import defaultdict

import pandas as pd
import numpy as np


def get_lang_confusion_for_one_model(dataset2langdist, file, step_=1):
    dataset2langdist= defaultdict(dict)

    with open(file) as f:
        model_log = json.load(f)
    # "../eval_logs/multilingual/eval_mt5_me5_arab-script_32_2layers_inverter.json"
    model_name = os.path.basename(file).replace("eval_", "").replace(".json", "")
    # model_name = os.path.basename(file).replace("eval_mt5_", "").replace(".json", "").replace("_32_2layers",
    #  "").replace("ara-script", "arab-script")
    # model_name = model_name.replace("_32", "").replace("_inverter", "").replace("_corrector", "")

    print(model_name)
    if model_name not in dataset2langdist:
        dataset2langdist[model_name] = dict()

    if "inverter" in file:

        for eval in model_log["evaluations"]:
            dataset = eval["dataset"]
            if dataset not in dataset2langdist[model_name]:
                dataset2langdist[model_name][dataset] = dict()

            lang_file = f"{eval['embeddings_file']}/eval_lang.json"
            if os.path.exists(lang_file):
                with open(lang_file) as f:
                    lang_eval = json.load(f)
                pred_lang_line = lang_eval["pred_lang_line_level_ratio"]
                true_lang_line = lang_eval["labels_lang_line_level_ratio"]
                pred_lang_line_dict = {k: v for k, v in pred_lang_line.items() if k not in ["unknown", "others"] and v > 0.05}
                true_lang_line_dict = {k: v for k, v in true_lang_line.items() if k not in ["unknown", "others"] and v > 0.05}
                print(dataset, "->", pred_lang_line_dict, "true:", true_lang_line_dict)

                if "labels" not in dataset2langdist[dataset]:
                    dataset2langdist[model_name][dataset]["Labels"] = true_lang_line_dict
                if "Base" not in dataset2langdist[model_name][dataset]:
                    dataset2langdist[model_name][dataset]["Base"] = pred_lang_line_dict

    if "corrector" in file:
        for dataset, evalsteps in model_log["evaluations"].items():
            if dataset not in dataset2langdist[model_name]:
                dataset2langdist[model_name][dataset] = dict()

            for step, evalresult in evalsteps.items():

                if step_ == 1 and step == "steps 1":
                    if "embeddings_files" in evalresult:
                        lang_file = f"{evalresult['embeddings_files']}/eval_lang.json"
                        if os.path.exists(lang_file):
                            with open(lang_file) as f:
                                lang_eval = json.load(f)
                            pred_lang_line = lang_eval["pred_lang_line_level_ratio"]
                            true_lang_line = lang_eval["labels_lang_line_level_ratio"]
                            pred_lang_line_dict = {k: v for k, v in pred_lang_line.items() if
                                                   k not in ["unknown", "others"] and v > 0.05}
                            true_lang_line_dict = {k: v for k, v in true_lang_line.items() if
                                                   k not in ["unknown", "others"] and v > 0.05}
                            print(dataset, "->", pred_lang_line_dict, "true:", true_lang_line_dict)

                            if "labels" not in dataset2langdist[model_name][dataset]:
                                dataset2langdist[model_name][dataset]["Labels"] = true_lang_line_dict
                            if "Step1" not in dataset2langdist[model_name][dataset]:
                                dataset2langdist[model_name][dataset]["Step1"] = pred_lang_line_dict

                if step.endswith("beam width 8") and step_ == 50:
                    if "embeddings_files" in evalresult:
                        lang_file = f"{evalresult['embeddings_files']}/eval_lang.json"

                        if os.path.exists(lang_file):
                            with open(lang_file) as f:
                                lang_eval = json.load(f)
                            pred_lang_line = lang_eval["pred_lang_line_level_ratio"]
                            true_lang_line = lang_eval["labels_lang_line_level_ratio"]
                            pred_lang_line_dict = {k: v for k, v in pred_lang_line.items() if
                                                   k not in ["unknown", "others"] and v > 0.05}
                            true_lang_line_dict = {k: v for k, v in true_lang_line.items() if
                                                   k not in ["unknown", "others"] and v > 0.05}
                            print(dataset, "->", pred_lang_line_dict, "true:", true_lang_line_dict)

                            if "labels" not in dataset2langdist[model_name][dataset]:
                                dataset2langdist[model_name][dataset]["Labels"] = true_lang_line_dict
                            if "Step50+sbeam8" not in dataset2langdist[model_name][dataset]:
                                dataset2langdist[model_name][dataset]["Step50+sbeam8"] = pred_lang_line_dict

                            break


def main():
    dataset2langdist = defaultdict(dict)

    for file in os.listdir("eval_logs/monolingual"):
        filepath = os.path.join("eval_logs/monolingual", file)
        if filepath.endswith(".json"):
            for step_ in [1, 50]:
                get_lang_confusion_for_one_model(dataset2langdist, filepath, step_=step_)

    for file in os.listdir("eval_logs/multilingual"):
        filepath = os.path.join("eval_logs/multilingual", file)
        if filepath.endswith(".json"):
            for step_ in [1, 50]:
                get_lang_confusion_for_one_model(dataset2langdist, filepath, step_=step_)

    with open("language_confusion/dataset2langdist.json", "w") as f:
        json.dump(dataset2langdist, f)


if __name__ == '__main__':
    main()
