import json
import os

import pandas as pd
from collections import defaultdict


def load_json_file(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def get_result_models(results_dict, step=1, inverter=False):
    models_result_dict = defaultdict(dict)

    for model, dataset_eval in results_dict.items():

        if inverter:
            if "inverter" in model:
                if model not in models_result_dict:
                    models_result_dict[model] = dict()
                for dataset, eval_result in dataset_eval.items():
                    models_result_dict[model][dataset] = eval_result["Base"]

        else:
            if "corrector" in model:
                if model not in models_result_dict:
                    models_result_dict[model] = dict()

                for dataset, eval_result in dataset_eval.items():
                    for step_, step_eval in eval_result.items():
                        if step == 1:
                            if step_ == "Step1":
                                models_result_dict[model][dataset] = step_eval
                        else:
                            if step_ == "Step50_sbeam8":
                                models_result_dict[model][dataset] = step_eval

    return models_result_dict


def processing_one_step(filepath, step=0, inverter=True):
    # loading file
    results_dict = load_json_file(filepath)
    models_results_dict = get_result_models(results_dict, step, inverter)
    # return dataframe with models as indices, and eval dataset as columns.
    return pd.DataFrame.from_records(models_results_dict).T


def processing_one_eval_df(df, filepath):
    """
    processing eval file from mt5_me5 models
    """
    model_list_inverter = ["me5_deu_Latn_inverter", "me5_heb_Hebr_inverter", "me5_cmn_Hani_inverter",
                           "me5_indo-aryan-fami_inverter", "me5_semitic-fami_inverter", "me5_turkic-fami_inverter",
                           "me5_atlatic_fami_inverter",
                           "me5_arab-script_inverter", "me5_cyrl-script_inverter", "me5_latn-script_inverter"]

    model_list_corrector = ['me5_deu_Latn_corrector', 'me5_heb_Hebr_corrector', 'me5_cmn_Hani_corrector',
                            'me5_indo-aryan-fami_corrector', 'me5_semitic-fami_corrector', 'me5_turkic-fami_corrector',
                            'me5_atlatic_fami_corrector',
                            'me5_latn-script_corrector', 'me5_ara-script_corrector', 'me5_cyrl-script_corrector']
    assert len(model_list_corrector) == 10
    assert len(model_list_inverter) == 10
    evals = [
        'deu_Latn', 'mlt_Latn', 'tur_Latn', 'hun_Latn', 'fin_Latn',
        'kaz_Cyrl', 'mhr_Cyrl', 'mon_Cyrl',
        'ydd_Hebr', 'heb_Hebr',
        'arb_Arab', 'urd_Arab',
        'hin_Deva', 'guj_Gujr', 'sin_Sinh', 'pan_Guru',
        'cmn_Hani', 'jpn_Jpan', 'kor_Hang',
        'amh_Ethi']
    if "multilingual_" in filepath:
        df.index = df.index.str.replace("yiyic/mt5_", "")
        df.index = df.index.str.replace("_32_2layers", "")
        df = df.reindex(columns=evals)
        if "inverter" in filepath:
            df = df.reindex(index=model_list_inverter)
        else:
            df = df.reindex(index=model_list_corrector)
    elif "monolingual_" in filepath:
        df.index = df.index.str.replace("yiyic/mt5_", "")
        df.index = df.index.str.replace("_32", "")
        df = df.reindex(columns=evals)

    return df


def processing_results_one_file(filepath):
    basename = os.path.basename(filepath).replace(".json", "")
    dirn = os.path.dirname(filepath)

    df_base = processing_one_step(filepath, 0, True)
    df_base = processing_one_eval_df(df_base, filepath)
    df_base.to_csv(os.path.join(dirn, f"{basename}_base_inverter.csv"))

    df_step1 = processing_one_step(filepath, 1, False)
    df_step1 = processing_one_eval_df(df_step1, filepath)
    df_step1.to_csv(os.path.join(dirn, f"{basename}_step1_corrector.csv"))

    df_step50_beam8 = processing_one_step(filepath, 50, False)
    df_step50_beam8 = processing_one_step(df_step50_beam8, filepath)
    df_step50_beam8.to_csv(os.path.join(dirn, f"{basename}_step50_sbeam8_corrector.csv"))


def processing_results_batch(folder="results/mt5_me5/"):
    for file in os.listdir(folder):
        filepath = os.path.join(folder, file)
        print(f"Processing filepath {filepath}")
        processing_results_one_file(filepath)


if __name__ == '__main__':
    import plac

    plac.call(processing_results_batch)
