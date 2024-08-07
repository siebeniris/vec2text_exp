import json
import pandas as pd
import numpy as np
from itertools import chain, combinations

from ast import literal_eval
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss) + 1)))


with open("language_confusion/data/languages.json") as f:
    languages = json.load(f)


def load_data(data_dir="language_confusion/data"):
    print(f"Loading data ...")

    X_test = pd.read_csv(f"{data_dir}/X_test.csv", index_col=0)
    X_train = pd.read_csv(f"{data_dir}/X_train.csv", index_col=0)
    y_test = pd.read_csv(f"{data_dir}/y_test.csv", index_col=0)
    y_train = pd.read_csv(f"{data_dir}/y_train.csv", index_col=0)

    return X_train, y_train, X_test, y_test


def RandomForest(X_train, y_train, X_test, y_test):
    print(f"RandomForestRegressor...")
    regr = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred, multioutput="raw_values")
    mae = mean_absolute_error(y_test, y_pred, multioutput="raw_values")

    langs = []
    mse_values_list = []
    mae_values_list = []
    for lang, mse_values, mae_values in zip(languages, mse, mae):
        # print(f"MSE for {lang}: {mse_values:.5f}")
        langs.append(lang)
        mse_values_list.append(mse_values)
        mae_values_list.append(mae_values)

    langs.append("avg")
    mse_values_list.append(np.mean(mse))
    mae_values_list.append(np.mean(mae))

    df_mse = pd.DataFrame.from_dict({
        "lang": langs,
        "mse": mse_values_list,
        "mae": mae_values_list

    })
    print("avg:", np.mean(mse), " | ", np.mean(mae))
    return df_mse


def run_regression_ablation(regressor="random_forest"):
    # eval_lang_encoded,script,family,script_lr,training_script_lr,
    # training lang and steps.
    #  arb_Arab,cmn_Hani,deu_Latn,guj_Gujr,heb_Hebr,hin_Deva,jpn_Jpan,kaz_Cyrl,mon_Cyrl,pan_Guru,tur_Latn,urd_Arab,step_Base,step_Step1,step_Step50+sbeam8

    # same script
    variables = ["script", "family", "script_lr", "training_script_lr", "emb_cos_sim"]

    all_combs = all_subsets(variables)

    for comb in all_combs:
        print(f"minus {comb}")
        X_train, y_train, X_test, y_test = load_data()

        remain_columns = sorted(list(set(list(variables)).difference(set(list(comb)))))
        print(f"remaining columns: {remain_columns}")

        if len(comb) > 0:
            X_train = X_train.drop(columns=list(comb))
            X_test = X_test.drop(columns=list(comb))

        if regressor == "random_forest":
            filename = f"language_confusion/results/random_forest/baseline+{','.join(remain_columns)}.csv"

            df_mse = RandomForest(X_train, y_train, X_test, y_test)
            df_mse.to_csv(filename, index=False)


if __name__ == '__main__':
    run_regression_ablation()
