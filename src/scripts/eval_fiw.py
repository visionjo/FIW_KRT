# This script evaluates features from the KinFaceW-(I/II) dataset
from __future__ import print_function
import numpy as np
from src.fiwtools.utils import features as futils
import glob
import sklearn.metrics.pairwise as pw
from sklearn.metrics import roc_curve, auc
import src.fiwtools.utils.io as io

# from src.common.utilities import *
import src.fiwtools.kinwild as kinwild
import sklearn.preprocessing as skpreprocess
import pandas as pd


from src.configs import CONFIGS

layers = ['conv5_2', 'conv5_3', 'pool5', 'fc6', 'fc7']
layers = ['res5a']
# lid = 4
# sub_dirs = ['father-dau', 'father-son',  'mother-dau', 'mother-son']

dir_root = CONFIGS.path.dbroot
dir_features = CONFIGS.path.dfeatures
dir_results = CONFIGS.path.doutput
io.mkdir(dir_results)

dir_pairs = dir_root + "Pairs/folds_5splits/"

# load experimental settings for 5-fold verification
f_lists = glob.glob(dir_pairs + "*.csv")
pair_types = [io.file_base(f).replace('-folds', '') for f in f_lists]

# dir_feats = [dir_features + p + "/" for p in pair_types]
import os
for i in range(0, 11):
    df_list = pd.read_csv(f_lists[i])
    pair_type = io.file_base(f_lists[i]).replace('-folds', '')
    labels = np.array(df_list.label)
    folds = np.array(df_list.fold)
    for layer in layers:
        fold_list = list(set(folds))
        dir_out = dir_results + pair_type + "/" + layer + "/"
        if os.path.isdir(dir_out) is True:
            do_continue = True
            for fold in fold_list:
                if not os.path.isdir(dir_out + str(fold) + "/"):
                    do_continue = False
            if do_continue:
                continue

        io.mkdir(dir_out)
        p1_list = [dir_features + layer + '/' + p.replace('.jpg', '.csv') for p in df_list.p1]
        p2_list = [dir_features + layer + '/' + p.replace('.jpg', '.csv') for p in df_list.p2]

        print("processing features from layer", layer)
        feats1 = kinwild.load_all_features("", p1_list)
        feats2 = kinwild.load_all_features("", p2_list)

        # dir_result = dir_out #dir_results + layer + "/"
        #io.mkdir(dir_results)
        ts_matches = []
        sim = []
        for fold in fold_list:
            print("Fold", fold, "/", len(fold_list))
            dir_fold = dir_out + str(fold) + "/"
            if os.path.isdir(dir_fold):
                continue
            train_ids = fold != folds
            test_ids = fold == folds

            train_labels = labels[train_ids]
            test_labels = labels[test_ids]
            # pairs 1
            train_feats1 = feats1[train_ids]
            test_feats1 = feats1[test_ids]

            # pairs 2
            train_feats2 = feats2[train_ids]
            test_feats2 = feats2[test_ids]

            feats = []
            for f1, f2 in zip(train_feats1[train_labels == 1], train_feats1[train_labels == 1]):
                feats.append(f1)
                feats.append(f2)

            # all_feats = {**tuple(trfeats1), **tuple(trfeats2)}
            print("Normalizing Features")
            feat_vecs = np.array(set(feats))
            std_scale = skpreprocess.StandardScaler().fit(feat_vecs)
            X_train = std_scale.transform(feat_vecs)

            print("PCA")
            # pca [default k is 200]
            X_scaled, evals, evecs = futils.pca(X_train)

            # normalize test set
            X_feats_1 = std_scale.transform(test_feats1)
            X_feats_2 = std_scale.transform(test_feats2)

            # apply PCA
            X_scaled_1 = np.dot(evecs.T, X_feats_1.T).T
            X_scaled_2 = np.dot(evecs.T, X_feats_2.T).T

            cosine_sim = pw.cosine_similarity(X_scaled_1, X_scaled_2)
            scores = np.diag(cosine_sim)

            # val_labels = [int(pair[1]) for pair in pair_val_list]

            ts_matches.append(test_labels)
            sim.append(scores)
            # Compute ROC curve and ROC area
            fpr, tpr, _ = roc_curve(test_labels, scores)
            roc_auc = auc(fpr, tpr)
            print("Acc:", roc_auc)
            print('Saving Results')

            io.mkdir(dir_fold)
            np.savetxt(dir_fold + "fpr.csv", fpr)
            np.savetxt(dir_fold + "tpr.csv", tpr)
            np.savetxt(dir_fold + "roc_auc.csv", [roc_auc])

        # fpr, tpr, _ = roc_curve(np.array(ts_matches).flatten(), np.array(sim).flatten())
        # roc_auc = auc(fpr, tpr)
        # np.savetxt(dir_results + pair_type + "/" + "fpr.csv", fpr)
        # np.savetxt(dir_results + pair_type + "/" + "tpr.csv", tpr)
        # np.savetxt(dir_results + pair_type + "/" + "roc_auc.csv", [roc_auc])