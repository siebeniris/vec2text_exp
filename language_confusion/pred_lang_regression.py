import json
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_data(data_dir="language_confusion/data"):

    X_test = pd.read_csv(f"{data_dir}/X_test.csv", index_col=0)
    X_train = pd.read_csv(f"{data_dir}/X_train.csv", index_col=0)
    y_test = pd.read_csv(f"{data_dir}/y_test.csv", index_col=0)
    y_train = pd.read_csv(f"{data_dir}/y_train.csv", index_col=0)



def RandomForest(X,y):

