# This script evaluates features from various layers of VGG of the faces from KinFaceW-(I/II) dataset
# TODO: make script with command line arguments (i.e., argparse)
import numpy as np
import glob
import sklearn.metrics.pairwise as pw
from sklearn.metrics import roc_curve, auc
import common.io as io
import database.kinwild as kinwild
import sklearn.preprocessing as skpreprocess
from sklearn.decomposition import TruncatedSVD


features = ['conv5_2', 'conv5_3', 'pool5', 'fc6', 'fc7']
sub_dirs = ['father-dau', 'father-son',  'mother-dau', 'mother-son']

dir_root = '/media/jrobby/Seagate Backup Plus Drive1/DATA/Kinship/KinFaceW-II/'
dir_features = dir_root + 'vgg_face/'

dir_results = dir_features + 'results_spca/'
io.mkdir(dir_results)

dir_perms = dir_root + 'perm/'
dir_lists = dir_root + 'meta_data/'

do_pca = True
k = 200
# load experimental settings for 5-fold verification
f_lists = glob.glob(dir_lists + "*.csv")
pair_types = [io.file_base(f) for f in f_lists]

dir_feats = [dir_features + p + "/" for p in sub_dirs]

fold_list = [1, 2, 3, 4, 5]
for ids in [1]:
    folds, labels, pairs1, pairs2 = kinwild.read_pair_list(f_lists[ids])

    d_out = dir_results + pair_types[ids] + "/"
    io.mkdir(d_out)

    for feature in features:
        print("processing features from layer", feature)
        feats1 = kinwild.load_all_features(dir_feats[ids] + feature + "/", pairs1)
        feats2 = kinwild.load_all_features(dir_feats[ids] + feature + "/", pairs2)
        dir_result = d_out + feature + "/"
        io.mkdir(dir_result)
        ts_matches = []
        sim = []
        accs = np.zeros((5,))
        for jj, fold in enumerate(fold_list):
            print("Fold", fold, "/", len(fold_list))
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
            feat_vecs = np.array(feats)
            std_scale = skpreprocess.StandardScaler().fit(feat_vecs)

            # normalize test set
            X_feats_1 = std_scale.transform(test_feats1)
            X_feats_2 = std_scale.transform(test_feats2)

            if do_pca:
                X_train = std_scale.transform(feat_vecs)

                print("PCA")

                try:
                    # pca [default k is 200]
                    clf = TruncatedSVD(k)
                    evecs = clf.fit_transform(X_train.T)
                except ValueError:
                    # if reduced dimensionality is less than or equal to the number of samples
                    knew = X_train.shape[0] - 1
                    clf = TruncatedSVD(knew)
                    evecs = clf.fit_transform(X_train.T)

                # X_scaled, evals, evecs = futils.pca(X_train)

                # apply PCA
                X_feats_1 = np.dot(evecs.T, X_feats_1.T).T
                X_feats_2 = np.dot(evecs.T, X_feats_2.T).T

            cosine_sim = pw.cosine_similarity(X_feats_1, X_feats_2)
            scores = np.diag(cosine_sim)

            # val_labels = [int(pair[1]) for pair in pair_val_list]

            ts_matches.append(test_labels)
            sim.append(scores)
            # Compute ROC curve and ROC area
            fpr, tpr, _ = roc_curve(test_labels, scores)
            roc_auc = auc(fpr, tpr)
            print("Acc:", roc_auc)
            print('Saving Results')
            dir_out = dir_result + str(fold) + "/"
            io.mkdir(dir_out)
            np.savetxt(dir_out + "fpr.csv", fpr)
            np.savetxt(dir_out + "tpr.csv", tpr)
            np.savetxt(dir_out + "roc_auc.csv", [roc_auc])
            accs[jj] = roc_auc
        fpr, tpr, _ = roc_curve(np.array(ts_matches).flatten(), np.array(sim).flatten())
        roc_auc = auc(fpr, tpr)
        np.savetxt(dir_result + "fpr.csv", fpr)
        np.savetxt(dir_result + "tpr.csv", tpr)
        np.savetxt(dir_result + "roc_auc.csv", [accs.mean()])
        np.savetxt(dir_result + "acc.csv", [accs.mean()])