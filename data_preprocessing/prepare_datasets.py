from time import sleep
from huggingface_hub import login as huggingface_login
import yaml
from itertools import combinations
from datasets import load_dataset, DatasetDict, concatenate_datasets


def login_huggingface():
    """login huggingface account for data."""
    access_token = "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"
    return huggingface_login(token=access_token)


login_huggingface()

with open("vec2text/lang2file.yaml", "r") as f:
    lang2file = yaml.safe_load(f)


def get_data_combos(lang1, lang2):
    lang1_train = load_dataset(lang2file[lang1])["train"]
    lang1_dev = load_dataset(lang2file[lang1].replace("_train", "_dev"))["train"].select(range(250))
    lang1_len = lang1_train.num_rows

    lang2_train = load_dataset(lang2file[lang2])["train"]
    lang2_dev = load_dataset(lang2file[lang2].replace("_train", "_dev"))["train"].select(range(250))
    lang2_len = lang2_train.num_rows

    print(f"processing train dataset {lang1} : {lang1_len} and {lang2}: {lang2_len}")

    min_row = min(lang1_len, lang2_len)
    if lang2_len > min_row:
        lang2_train = lang2_train.select(range(min_row))
    if lang1_len > min_row:
        lang1_train = lang1_train.select(range(min_row))

    lang1_lang2_train= concatenate_datasets([lang1_train, lang2_train])
    lang1_lang2_train.push_to_hub(f"yiyic/{lang1}_{lang2}_train")

    lang1_lang2_dev= concatenate_datasets([lang1_dev, lang2_dev])
    lang1_lang2_dev.push_to_hub(f"yiyic/{lang1}_{lang2}_dev")


def main():
    indo_aryans = ["hin_Deva", "guj_Gujr", "pan_Guru", "urd_Arab"]
    turkic = ["kaz_Cyrl", "tur_Latn"]
    semitic = ("heb_Hebr", "arb_Arab")
    combos_indo_aryans = list(combinations(indo_aryans, 2))
    combos_random = list(set([(x,y) for x in turkic for y in indo_aryans]))

    for t1, t2 in semitic:
        get_data_combos(t1, t2)

    for combo in combos_indo_aryans:
        x,y = combo
        get_data_combos(x,y)

    for combo in combos_random:
        x,y = combo
        get_data_combos(x,y)

if __name__ == '__main__':
    main()
