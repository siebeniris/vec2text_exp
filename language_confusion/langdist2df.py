import pandas as pd
import json
import os

from ast import literal_eval
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

evals = [
    'deu_Latn', 'mlt_Latn', 'tur_Latn', 'hun_Latn', 'fin_Latn',
    'kaz_Cyrl', 'mhr_Cyrl', 'mon_Cyrl',
    'ydd_Hebr', 'heb_Hebr',
    'arb_Arab', 'urd_Arab',
    'hin_Deva', 'guj_Gujr', 'sin_Sinh', 'pan_Guru',
    'cmn_Hani', 'jpn_Jpan', 'kor_Hang',
    'amh_Ethi']

lang2langscript = {x.split("_")[0]: x for x in evals}

with open("language_confusion/dataset2langdist.json") as f:
    langdist = json.load(f)


def get_key2lang(model2langs):
    key2lang = {}
    for key in model2langs:
        if "_" in key:
            if len(key) == 7:
                lang1, lang2 = key.split("_")
                key2lang[key] = [lang2langscript[lang1], lang2langscript[lang2]]
            if len(key) == 8:
                key2lang[key] = [key]
        elif key == "turkic-fami":
            key2lang[key] = ['tur_Latn', 'kaz_Cyrl']

        elif key == "arab-script":
            key2lang[key] = ["arb_Arab", "heb_Hebr"]
        elif key == "latn-script":
            key2lang[key] = ["deu_Latn", "tur_Latn"]
    return key2lang


def get_cos_similarity(filepath="results/mt5_me5/multilingual_eval_emb_cos_sim.json"):
    # get cosine similarities, to see if it can help predict languages
    with open(filepath) as f:
        data = json.load(f)
    model_names = []
    steps = []
    values = []
    eval_langs = []
    for model, eval_dict in data.items():
        model_name = model.replace("yiyic/mt5_me5_", "").replace("_32_2layers", "").replace("_corrector", "").replace(
            "_inverter", "")
        for eval_lang, eval_steps in eval_dict.items():
            for step, value in eval_steps.items():
                if step == "Step50_sbeam8":
                    steps.append("Step50+sbeam8")
                else:
                    steps.append(step)
                values.append(value)
                model_names.append(model_name)
                eval_langs.append(eval_lang)
    df = pd.DataFrame.from_dict({
        "model": model_names,
        "eval_lang": eval_langs,
        "step": steps,
        "emb_cos_sim": values
    })
    return df


def get_model2langs(langdist):
    model2langs = dict()
    for model, langs_dict in langdist.items():
        if "me5_" in model:
            model_name = model.replace("mt5_me5_", "").replace("_32_2layers_corrector", "").replace(
                "_32_2layers_inverter",
                "")
            # print(model_name)

            if model_name not in model2langs:
                if model_name == "ara-script":
                    model2langs["arab-script"] = dict()
                    model_name = "arab-script"
                else:
                    model2langs[model_name] = dict()

            for eval_lang, steps_eval in langs_dict.items():
                if eval_lang not in model2langs[model_name]:
                    model2langs[model_name][eval_lang] = steps_eval
                for step, eval in steps_eval.items():
                    if step not in model2langs[model_name][eval_lang]:
                        model2langs[model_name][eval_lang] = dict()
                    model2langs[model_name][eval_lang][step] = eval
    # output to model2langs.json
    key2lang = get_key2lang(model2langs)

    model2langs_dict = dict()

    for key, langs in key2lang.items():
        if key not in model2langs_dict:
            model2langs_dict[key] = dict()
        model2langs_dict[key]["training"] = langs
        model2langs_dict[key]["langdict"] = model2langs[key]

    training_data_list = []  # [cmn_Hani, jpn_Jpan]
    eval_langs = []  # deu_Latn
    steps = []  # base, step1, step50+sbeam8
    pred_langs = []  # {'deu_Latn': 1.0}
    inversion_models = []

    for model in model2langs_dict:
        training_data = model2langs_dict[model]["training"]
        langdict = model2langs_dict[model]["langdict"]

        for eval_lang, step_eval in langdict.items():
            # deu_latn, xxxx
            for step, eval in step_eval.items():
                inversion_models.append(model)
                training_data_list.append(training_data)
                eval_langs.append(eval_lang)
                steps.append(step)
                pred_langs.append(eval)

    df_lang = pd.DataFrame({"model": inversion_models, "training": training_data_list,
                            "eval_lang": eval_langs, "step": steps,
                            "pred_langs": pred_langs})
    df_cos = get_cos_similarity()

    df_lang_ = pd.merge(df_lang, df_cos, on=["model", "step", "eval_lang"], how="left")
    print(len(df_lang_))
    df_lang_.to_csv("language_confusion/model2langs.csv", index=False)
    return df_lang_


if __name__ == '__main__':
    get_model2langs(langdist)
