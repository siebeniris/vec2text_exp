import json
import os

import pandas as pd
from collections import defaultdict

evals = [
    'deu_Latn', 'mlt_Latn', 'tur_Latn', 'hun_Latn', 'fin_Latn',
    'kaz_Cyrl', 'mhr_Cyrl', 'mon_Cyrl',
    'ydd_Hebr', 'heb_Hebr',
    'arb_Arab', 'urd_Arab',
    'hin_Deva', 'guj_Gujr', 'sin_Sinh', 'pan_Guru',
    'cmn_Hani', 'jpn_Jpan', 'kor_Hang',
    'amh_Ethi']

lang2eval = {x.split("_")[0]: x for x in evals}


def gather_results_monolingual(results_dir="results/mt5_me5"):
    """
    Gather results in family on monolingual settings.
    """
    results_dict = defaultdict(dict)
    inverters = [
        'mt5_me5_deu_Latn_32_2layers_inverter', 'mt5_me5_heb_Hebr_32_2layers_inverter',
        'mt5_me5_cmn_Hani_32_2layers_inverter',
        'mt5_me5_arb_Arab_32_2layers_inverter', 'mt5_me5_jpn_Jpan_32_2layers_inverter',
        'mt5_me5_tur_Latn_32_2layers_inverter', 'mt5_me5_kaz_Cyrl_32_2layers_inverter',
        'mt5_me5_mon_Cyrl_32_2layers_inverter', 'mt5_me5_urd_Arab_32_2layers_inverter',
        'mt5_me5_pan_Guru_32_2layers_inverter', 'mt5_me5_guj_Gujr_32_2layers_inverter',
        'mt5_me5_hin_Deva_32_2layers_inverter']

    correctors = [
        'mt5_me5_deu_Latn_32_2layers_corrector', 'mt5_me5_heb_Hebr_32_2layers_corrector',
        'mt5_me5_cmn_Hani_32_2layers_corrector',
        'mt5_me5_arb_Arab_32_2layers_corrector', 'mt5_me5_jpn_Jpan_32_2layers_corrector',
        'mt5_me5_tur_Latn_32_2layers_corrector', 'mt5_me5_kaz_Cyrl_32_2layers_corrector',
        'mt5_me5_mon_Cyrl_32_2layers_corrector', 'mt5_me5_urd_Arab_32_2layers_corrector',
        'mt5_me5_pan_Guru_32_2layers_corrector', 'mt5_me5_guj_Gujr_32_2layers_corrector',
        'mt5_me5_hin_Deva_32_2layers_corrector']

    for step in ["base_inverter", "step1_corrector", "step50_sbeam8_corrector"]:

        for metric in ["num_true_words", "num_pred_words", "token_set_f1", "bleu_score", "rouge_score", "emb_cos_sim"]:
            filepath = f"{results_dir}/multilingual_eval_{metric}_{step}.csv"
            df = pd.read_csv(filepath, index_col=0)

            if "inverter" in filepath:
                for inverter in inverters:
                    lang = inverter.replace("mt5_me5_", "").replace("_32_2layers_inverter", "")
                    lang_step = f"{lang}_{step}"

                    if lang_step not in results_dict:
                        results_dict[lang_step] = dict()

                    value = df.loc[inverter, lang]
                    results_dict[lang_step][metric] = value

                    # results_dict[lang_step]["step"] = step.replace("_inverter", "")
                    # results_dict[lang_step]["eval_lang"] = lang

            if "corrector" in filepath:
                for corrector in correctors:
                    lang = corrector.replace("mt5_me5_", "").replace("_32_2layers_corrector", "")
                    lang_step = f"{lang}_{step}"
                    if lang_step not in results_dict:
                        results_dict[lang_step] = dict()

                    value = df.loc[corrector, lang]
                    results_dict[lang_step][metric] = value

                    # results_dict[lang_step]["step"] = step.replace("_corrector", "")
                    # results_dict[lang_step]["eval_lang"] = lang

    df = pd.DataFrame.from_records(results_dict).T
    print(df.head())
    df.to_csv(f"{results_dir}/processed/me5-mono.csv")


def gather_results_from_random_pairs(results_dir="results/mt5_me5"):
    """
    Gather results from random pairs in Indo-Aryan and Turkic language families.
    """
    results_dict = defaultdict(dict)
    inverters = [
        'mt5_me5_tur_urd_32_2layers_inverter',
        'mt5_me5_tur_pan_32_2layers_inverter', 'mt5_me5_tur_guj_32_2layers_inverter',
        'mt5_me5_tur_hin_32_2layers_inverter', 'mt5_me5_kaz_urd_32_2layers_inverter',
        'mt5_me5_kaz_pan_32_2layers_inverter', 'mt5_me5_kaz_guj_32_2layers_inverter',
        'mt5_me5_kaz_hin_32_2layers_inverter']

    correctors = [
        'mt5_me5_tur_urd_32_2layers_corrector',
        'mt5_me5_tur_pan_32_2layers_corrector', 'mt5_me5_tur_guj_32_2layers_corrector',
        'mt5_me5_tur_hin_32_2layers_corrector', 'mt5_me5_kaz_urd_32_2layers_corrector',
        'mt5_me5_kaz_pan_32_2layers_corrector', 'mt5_me5_kaz_guj_32_2layers_corrector',
        'mt5_me5_kaz_hin_32_2layers_corrector'
    ]

    for step in ["base_inverter", "step1_corrector", "step50_sbeam8_corrector"]:
        for metric in ["num_true_words", "num_pred_words", "token_set_f1", "bleu_score", "rouge_score", "emb_cos_sim"]:
            filepath = f"{results_dir}/multilingual_eval_{metric}_{step}.csv"
            df = pd.read_csv(filepath, index_col=0)

            if "inverter" in filepath:
                for inverter in inverters:
                    script = inverter.replace("mt5_me5_", "").replace("_32_2layers_inverter", "")
                    print(script)
                    lang1, lang2 = script.split("_")
                    langs = [lang2eval[lang] for lang in [lang1, lang2]]
                    for lang in langs:
                        script_lang_step = f"{script}_{lang}_{step}"
                        if script_lang_step not in results_dict:
                            results_dict[script_lang_step] = dict()

                        value = df.loc[inverter, lang]
                        results_dict[script_lang_step][metric] = value

            if "corrector" in filepath:
                for corrector in correctors:
                    script = corrector.replace("mt5_me5_", "").replace("_32_2layers_corrector", "")
                    lang1, lang2 = script.split("_")
                    langs = [lang2eval[lang] for lang in [lang1, lang2]]
                    for lang in langs:
                        script_lang_step = f"{script}_{lang}_{step}"
                        if script_lang_step not in results_dict:
                            results_dict[script_lang_step] = dict()

                        value = df.loc[corrector, lang]
                        results_dict[script_lang_step][metric] = value

    df = pd.DataFrame.from_records(results_dict).T
    print(df.head())
    df.to_csv(f"{results_dir}/processed/me5-random.csv")


def gather_results_from_in_family_pairs(results_dir="results/mt5_me5"):
    results_dict = defaultdict(dict)
    inverters = [
        'mt5_me5_heb_arb_32_2layers_inverter', 'mt5_me5_turkic-fami_32_2layers_inverter',
        'mt5_me5_urd_pan_32_2layers_inverter', 'mt5_me5_urd_guj_32_2layers_inverter',
        'mt5_me5_urd_hin_32_2layers_inverter', 'mt5_me5_hin_pan_32_2layers_inverter',
        'mt5_me5_hin_guj_32_2layers_inverter', 'mt5_me5_pan_guj_32_2layers_inverter']

    correctors = [
        'mt5_me5_heb_arb_32_2layers_corrector',
        'mt5_me5_turkic-fami_32_2layers_corrector',
        'mt5_me5_urd_pan_32_2layers_corrector', 'mt5_me5_urd_guj_32_2layers_corrector',
        'mt5_me5_urd_hin_32_2layers_corrector', 'mt5_me5_hin_pan_32_2layers_corrector',
        'mt5_me5_hin_guj_32_2layers_corrector', 'mt5_me5_pan_guj_32_2layers_corrector'
    ]

    for step in ["base_inverter", "step1_corrector", "step50_sbeam8_corrector"]:

        for metric in ["num_true_words", "num_pred_words", "token_set_f1", "bleu_score", "rouge_score", "emb_cos_sim"]:
            step_metric = f"{step}_{metric}"
            filepath = f"../results/mt5_me5/multilingual_eval_{metric}_{step}.csv"
            df = pd.read_csv(filepath, index_col=0)

            if "inverter" in filepath:
                for inverter in inverters:
                    script = inverter.replace("mt5_me5_", "").replace("_32_2layers_inverter", "")
                    if len(script) == 7:
                        lang1, lang2 = script.split("_")
                        langs = [lang2eval[lang] for lang in [lang1, lang2]]
                        for lang in langs:
                            script_lang_step = f"{script}_{lang}_{step}"
                            if script_lang_step not in results_dict:
                                results_dict[script_lang_step] = dict()

                            value = df.loc[inverter, lang]
                            results_dict[script_lang_step][metric] = value

                    elif script == "turkic-fami":
                        for lang in ["tur_Latn", "kaz_Cyrl"]:
                            script_lang_step = f"{script}_{lang}_{step}"
                            if script_lang_step not in results_dict:
                                results_dict[script_lang_step] = dict()

                            value = df.loc[inverter, lang]
                            results_dict[script_lang_step][metric] = value

            if "corrector" in filepath:
                print(filepath)
                for corrector in correctors:
                    script = corrector.replace("mt5_me5_", "").replace("_32_2layers_corrector", "")

                    if len(script) == 7:
                        lang1, lang2 = script.split("_")
                        langs = [lang2eval[lang] for lang in [lang1, lang2]]
                        for lang in langs:
                            script_lang_step = f"{script}_{lang}_{step}"
                            if script_lang_step not in results_dict:
                                results_dict[script_lang_step] = dict()

                            value = df.loc[corrector, lang]
                            results_dict[script_lang_step][metric] = value

                    elif script == "turkic-fami":
                        for lang in ["tur_Latn", "kaz_Cyrl"]:
                            script_lang_step = f"{script}_{lang}_{step}"
                            if script_lang_step not in results_dict:
                                results_dict[script_lang_step] = dict()

                            value = df.loc[corrector, lang]
                            results_dict[script_lang_step][metric] = value

    df = pd.DataFrame.from_records(results_dict).T
    df.to_csv(f"{results_dir}/processed/me5-family.csv")


def gather_results_from_in_script(results_dir="results/mt5_me5"):
    """

    """
    from collections import defaultdict

    results_dict = defaultdict(dict)
    inverters = [
        'mt5_me5_arab-script_32_2layers_inverter',
        'mt5_me5_latn-script_32_2layers_inverter',
        'mt5_me5_cyrl-script_32_2layers_inverter',
        'mt5_me5_cmn_jpn_32_2layers_inverter']

    correctors = [
        'mt5_me5_ara-script_32_2layers_corrector',
        'mt5_me5_latn-script_32_2layers_corrector',
        'mt5_me5_cyrl-script_32_2layers_corrector',
        'mt5_me5_cmn_jpn_32_2layers_corrector']

    for step in ["base_inverter", "step1_corrector", "step50_sbeam8_corrector"]:
        for metric in ["num_true_words", "num_pred_words", "token_set_f1", "bleu_score", "rouge_score", "emb_cos_sim"]:

            filepath = f"f{results_dir}/multilingual_eval_{metric}_{step}.csv"
            df = pd.read_csv(filepath, index_col=0)

            if "inverter" in filepath:
                for inverter in inverters:
                    script = inverter.replace("mt5_me5_", "").replace("_32_2layers_inverter", "")
                    if script == "arab-script":
                        for lang in ["arb_Arab", "urd_Arab"]:
                            script_lang_step = f"{script}_{lang}_{step}"
                            if script_lang_step not in results_dict:
                                results_dict[script_lang_step] = dict()

                            value = df.loc[inverter, lang]
                            results_dict[script_lang_step][metric] = value

                    if script == "latn-script":
                        for lang in ["deu_Latn", "tur_Latn"]:
                            script_lang_step = f"{script}_{lang}_{step}"
                            if script_lang_step not in results_dict:
                                results_dict[script_lang_step] = dict()

                            value = df.loc[inverter, lang]
                            results_dict[script_lang_step][metric] = value

                    if script == "cyrl-script":
                        for lang in ["kaz_Cyrl", "mon_Cyrl"]:
                            script_lang_step = f"{script}_{lang}_{step}"
                            if script_lang_step not in results_dict:
                                results_dict[script_lang_step] = dict()

                            value = df.loc[inverter, lang]
                            results_dict[script_lang_step][metric] = value

                    if script == "cmn_jpn":
                        for lang in ["cmn_Hani", "jpn_Jpan"]:
                            script_lang_step = f"{script}_{lang}_{step}"
                            if script_lang_step not in results_dict:
                                results_dict[script_lang_step] = dict()

                            value = df.loc[inverter, lang]
                            results_dict[script_lang_step][metric] = value

            if "corrector" in filepath:
                for corrector in correctors:
                    script = corrector.replace("mt5_me5_", "").replace("_32_2layers_corrector", "")
                    if script == "ara-script":
                        for lang in ["arb_Arab", "urd_Arab"]:
                            script_lang_step = f"{script}_{lang}_{step}"
                            if script_lang_step not in results_dict:
                                results_dict[script_lang_step] = dict()

                            value = df.loc[corrector, lang]
                            results_dict[script_lang_step][metric] = value

                    if script == "latn-script":
                        for lang in ["deu_Latn", "tur_Latn"]:
                            script_lang_step = f"{script}_{lang}_{step}"
                            if script_lang_step not in results_dict:
                                results_dict[script_lang_step] = dict()

                            value = df.loc[corrector, lang]
                            results_dict[script_lang_step][metric] = value

                    if script == "cyrl-script":
                        for lang in ["kaz_Cyrl", "mon_Cyrl"]:
                            script_lang_step = f"{script}_{lang}_{step}"
                            if script_lang_step not in results_dict:
                                results_dict[script_lang_step] = dict()

                            value = df.loc[corrector, lang]
                            results_dict[script_lang_step][metric] = value

                    if script == "cmn_jpn":
                        for lang in ["cmn_Hani", "jpn_Jpan"]:
                            script_lang_step = f"{script}_{lang}_{step}"
                            if script_lang_step not in results_dict:
                                results_dict[script_lang_step] = dict()

                            value = df.loc[corrector, lang]
                            results_dict[script_lang_step][metric] = value

    df = pd.DataFrame.from_records(results_dict).T
    print(df.head())
    df.to_csv(f"{results_dir}/processed/me5-script.csv")


def main(results_type='monolingual'):
    if results_type == "monolingual":
        gather_results_monolingual()
    elif results_type == "script":
        gather_results_from_in_script()
    elif results_type == "family":
        gather_results_from_in_family_pairs()
    elif results_type == "random":
        gather_results_from_random_pairs()
    else:
        print("choose [monolingual, script, family, random].")


if __name__ == '__main__':
    import plac

    plac.call(main)
