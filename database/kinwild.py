import pandas as pd
import numpy as np


def read_feature(f_feature):
    return np.genfromtxt(f_feature, delimiter=',')


def load_all_features(dir_features, f_features):
    """
    Loads set of features.
    :param dir_features: root directory containing features.
    :param f_features: file paths of features relative to dir_features
    :return: set of features
    :rtype: dict
    """
    feats = []
    for file in f_features:
        feats.append(read_feature(dir_features + file))

    return np.array(feats)


def read_pair_list(f_csv):
    pair_list = pd.read_csv(f_csv,delimiter=',', header=None)

    folds = np.array(pair_list[0])
    labels = np.array(pair_list[1])
    pairs1 = pair_list[2].values
    pairs1 = [p.replace(".jpg", ".csv").strip() for p in pairs1]
    pairs2 = pair_list[3].values
    pairs2 = [p.replace(".jpg", ".csv").strip() for p in pairs2]
    return folds, labels, pairs1, pairs2
