import glob
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import common.io as io
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split FIW pair in k folds.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input',
                        default=io.sys_home() + "/Dropbox/Families_In_The_Wild/Database/Pairs/fs-faces.csv",
                        help='Input file or list of files.')
    parser.add_argument('-o', '--output',
                        default=io.sys_home() + "/Dropbox/Families_In_The_Wild/Database/Pairs/fs-splits.csv",
                        help='File in which output is stored')
    parser.add_argument('-n', '--nfolds', default=5, help="Number of folds to split set into.")
    args = parser.parse_args()

    n_folds = args.nfolds
    # dir_face_pairs = "/Users/josephrobinson/Dropbox/Families_In_The_Wild/Database/Pairs/"

    # files_pairs = glob.glob(dir_face_pairs + '*-faces.csv')

    kf = KFold(n_splits=n_folds, shuffle=True)

    df_pairs = pd.read_csv(args.input)

    fids = list(set([fid[:5] for fid in list(df_pairs.p1)]))

    k_ids = kf.split(fids)

    folds = []
    fold = 1
    for _, ids in k_ids:
        kth_fids = np.sort(list(np.array(fids)[ids]))
        folds.append((fold, kth_fids))
        print(kth_fids)
        fold = fold + 1


    df_splits = []
    for fold, kth_fids in folds:
        split = []
        for fid in list(kth_fids):
            ids = [i for i, s in enumerate(df_pairs.p1) if str(fid) in s]
            split.append(df_pairs.loc[ids])
            # df_splits. append(np.where(str(fid) in list(df_pairs.p1)))
        df_split = pd.concat(split)
        fold_ids = np.zeros(len(df_split)) + fold

        pd.concat([pd.DataFrame(fold_ids, index=fold_ids), df_split], axis=1)

        df_splits.append(pd.concat(pd.DataFrame(fold_ids, index=fold_ids), df_split))

