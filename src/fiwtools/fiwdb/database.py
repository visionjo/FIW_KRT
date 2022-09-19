"""
Module of utilities for handling FIW DB.

Methods to download PIDs using URL, load Anns and LUTs, along with metadata (e.g., gender, mids, pair lists).
"""
from __future__ import print_function

import glob
import pandas as pd
import numpy as np
<<<<<<< HEAD:src/fiwdb/database.py
import common.image as imutils
import common.io as io
import common.log as log
=======
import fiwtools.utils.image as imutils
import fiwtools.utils.io as io
import fiwtools.utils.log as log
>>>>>>> master:src/fiwtools/fiwdb/database.py
import operator
import csv
import random
from sklearn.model_selection import KFold


# TODO urllib.request to handle thrown exceptions <p>Error: HTTP Error 403: Forbidden</p>
# TODO modify fold2set with optional args that spefify which fold merges into which set (i.e., currently hard coded).

logger = log.setup_custom_logger(__name__, f_log='fiwdb.log', level=log.INFO)
logger.info('FIW-DB')

dir_db = io.sys_home() + "/Dropbox/Families_In_The_Wild/Database/"
# dir_db = "/data/"
dir_fid_root = dir_db + "FIDs_NEW/"
# dir_fid_root = dir_db + "journal_data/FIDs/"


def download_images(f_pid_csv=dir_db + "FIW_PIDs_new.csv", dir_out=dir_db + "fiwimages/"):
    """
    Download FIW database by referencing PID LUT. Each PID is listed with corresponding URL. URL is downloaded and
    saved as <FID>/PID.jpg
    :type f_pid_csv: object
    :type dir_out: object
    """

    logger.info(
        f"FIW-DB-- Download_images!\n Source: {f_pid_csv}\n Destination: {dir_out}"
    )

    # load urls (image location), pids (image name), and fids (output subfolder)
    df_pid = load_pid_lut(str(f_pid_csv))

    df_io = df_pid[['FIDs', 'PIDs', 'URL']]

    logger.info(f"{int(df_io.count().mean())} photos to download")

    for i, img_url in enumerate(df_io['URL']):
        try:
            f_out = str(dir_out) + df_io['FIDs'][i] + "/" + df_io['PIDs'][i] + ".jpg"
            img = imutils.url_to_image(img_url)
            logger.info(f"Downloading {df_io['PIDs'][i]}\n{img_url}\n")
            imutils.saveimage(f_out, img)
        except Exception as e0:
            logger.error(f"Error with {df_io['PIDs'][i]}\n{img_url}\n")
            error_message = "<p>Error: %s</p>\n" % str(e0)
            logger.error(error_message)


def get_unique_pairs(ids_in):
    """

    :param ids_in:
    :return:
    """
    ids = [(p1, p2) if p1 < p2 else (p2, p1) for p1, p2 in zip(list(ids_in[0]), list(ids_in[1]))]
    return list(set(ids))


def load_rid_lut(f_csv=dir_db + "FIW_RIDs.csv"):
    """

    :param f_csv:
    :return:
    """

    return pd.read_csv(f_csv, delimiter=',')


def load_pid_lut(f_csv=dir_db + "FIW_PIDs_new.csv"):
    """

    :param f_csv:
    :return:
    """
    return pd.read_csv(f_csv, delimiter='\t')


def load_fid_lut(f_csv=dir_db + "FIW_FIDs.csv"):
    """
    Load FIW_FIDs.csv-- FID- Surname LUT
    :param f_csv:
    :return:
    """
    return pd.read_csv(f_csv, delimiter='\t')


def load_fids(dirs_fid=dir_fid_root):
    """
    Function loads fid directories and labels.
    :param dirs_fid: root folder containing FID directories (i.e., F0001-F1000)
    :return: (list, list):  (fid filepaths and fid labels of these)
    """
    dirs = glob.glob(dirs_fid + 'F????/')
    fid_list = [d[-6:-1] for d in dirs]

    return dirs, fid_list


def load_mids(dirs_fid, f_csv='mid.csv'):
    """
    Load CSV file containing member information, i.e., {MID : ID, Name, Gender}
    :type f_csv:        file name of CSV files containing member labels
    :param dirs_fid:    root folder containing FID/MID/ folders of DB.

    :return:
    """

    return [pd.read_csv(d + "/" + f_csv) for d in dirs_fid]


def load_relationship_matrices(dirs_fid=dir_fid_root, f_csv='mid.csv'):
    """
    Load CSV file containing member information, i.e., {MID : ID, Name, Gender}
    :type f_csv:        file name of CSV files containing member labels
    :param dirs_fid:    root folder containing FID/MID/ folders of DB.

    :return:
    """
    df_relationships = [pd.read_csv(d + f_csv) for d in dirs_fid]

    for i in range(len(df_relationships)):
        # df_relationships = [content.ix[:, 1:len(content) + 1] for content in df_rel_contents]

        df_relationships[i].index = range(1, len(df_relationships[i]) + 1)
        df_relationships[i] = df_relationships[i].ix[:, 1:len(df_relationships[i]) + 1]

    return df_relationships

def get_relationship_dictionaries(dir_fid,fids, f_csv='mid.csv'):
    """
    Load CSV file containing member information, i.e., {MID : ID, Name, Gender}
    :type f_csv:        file name of CSV files containing member labels
    :param dirs_fid:    root folder containing FID/MID/ folders of DB.

    :return:
    """

    dict_relationships = {}
    for fid in fids:
        df_relationships = pd.read_csv(dir_fid + '/' + fid + '/' + f_csv)
        df_relationships.index = range(1, len(df_relationships) + 1)
        df_relationships = df_relationships.ix[:, 1:len(df_relationships) + 1]
        dict_relationships[fid] = df_relationships
    return dict_relationships

def get_names_dictionaries(dir_fid,fids, f_csv='mid.csv'):
    """
    Load CSV file containing member information, i.e., {MID : ID, Name, Gender}
    :type f_csv:        file name of CSV files containing member labels
    :param dirs_fid:    root folder containing FID/MID/ folders of DB.

    :return:
    """

    dict_names = {}
    for fid in fids:
        df_relationships = pd.read_csv(dir_fid + '/' + fid + '/' + f_csv)
        dict_names[fid] = list(df_relationships['Name'])

    return dict_names
def parse_relationship_matrices(df_mid):
    """
    Parses out relationship matrix from MID dataframe.

    :return:
    """
    # df_relationships = [content.ix[:, 1:len(content) + 1] for content in df_rel_contents]
    df_relationships = df_mid
    df_relationships.index = range(1, len(df_relationships) + 1)
    df_relationships = df_relationships.ix[:, 1:len(df_relationships) + 1]

    return np.array(df_relationships)


def set_pairs(mylist, ids_in, kind, fid):
    """
    Adds items to mylist of unique pairs.
    :param mylist:
    :param ids_in:
    :param kind:
    :param fid:
    :return:
    """
    ids = get_unique_pairs(ids_in)
    for i in enumerate(ids):
        print(i)
        indices = list(np.array(i[1]) + 1)
        mylist.append(Pair(mids=indices, fid=fid, kind=kind))
        del indices
    return mylist


def split_families(fids, nfolds=5, shuffle=True, seed=123):
    kf = KFold(nfolds, shuffle=False, random_state=seed)
    if shuffle:
        fids = random.shuffle(fids)
    # return kf.get_n_splits(fids)

    shuffle=True
    kf = KFold(5, shuffle=shuffle, random_state=123)
    for train, test in kf.split(fids):
        print(f"{train} {test}")




def specify_gender(rel_mat, genders, gender):
    """
    :param rel_mat:
    :param genders: list of genders
    :param gender:  gender to search for {'Male' or Female}
    :type gender:   str
    :return:
    """
    ids_not=[]
    for j, s in enumerate(genders):
        print(j, s)
        if gender not in s:
            ids_not.append(j)
    # ids_not = [j for j, s in enumerate(genders) if gender not in s]
    # rel_mat[ids_not, :] = 0
    # rel_mat[:, ids_not] = 0

    return rel_mat


def folds_to_sets(f_csv=dir_db + 'journal_data/Pairs/folds_5splits/', dir_out=dir_db + "journal_data/Pairs/sets/"):
    """ Method used to merge 5 fold splits into 3 sets for RFIW (train, val, and test)"""

    f_in = glob.glob(f_csv + '*-folds.csv')

    for file in f_in:
        # each list of pairs <FOLD, LABEL, PAIR_1, PAIR_2>
        f_name = io.file_base(file)
        print(f"\nProcessing {f_name}\n")

        df_pairs = pd.read_csv(file)

        # merge to form train set
        df_train = df_pairs[(df_pairs['fold'] == 1) | (df_pairs['fold'] == 5)]
        df_train.to_csv(dir_out + "train/" + f_name.replace("-folds", "-train") + ".csv")

        # merge to form val set
        df_val = df_pairs[(df_pairs['fold'] == 2) | (df_pairs['fold'] == 4)]
        df_val.to_csv(dir_out + "val/" + f_name.replace("-folds", "-val") + ".csv")

        # merge to form test set
        df_test = df_pairs[(df_pairs['fold'] == 3)]
        df_test.to_csv(dir_out + "test/" + f_name.replace("-folds", "-test") + ".csv")

        # print stats
        print(
            f"{df_train['fold'].count()} Training;\t {df_val['fold'].count()} Val;\t{df_test['fold'].count()} Test"
        )


def parsing_families(f_csv='mid.csv'):
    import pdb
    pdb.set_trace()
    fid_list = load_fids(dir_fid_root)[0]
    fid_list.sort()
    tup_mid = [(d[-6:-1], pd.read_csv(d + "/" + f_csv)) for d in fid_list]

    nmembers = [int(mid[1].count().mean()) for mid in tup_mid]

    ids = np.array(nmembers).__lt__(3)
    df_mid = [(mid[1], mid[0]) for mid in tup_mid]
    # df_vals = [df[0].values[:, 0:-2] for df in df_mid]

    fam_list = []
    # df2 = pd.DataFrame(columns=('FID','MID','val', 'nrel'))
    fam_list2 = []
    tr_mids = []
    for df in df_mid:
        # vals = [d[0].values[:, 0:-2] for d in df]
        vals = df[0].values[:, 1:-2]
        # vals = np.zeros_like()
        votes = np.zeros_like(vals)
        # vals=df.values[:, 0:-2]
        votes[(vals == 4) | (vals == 1)] += 3
        votes[(vals == 2)] += 2
        votes[(vals == 6) | (vals == 3)] += 1
        # votes[((df.values[:,0:-2]) == 4) | ((df.values[:,0:-2]) == 1)] += 3
        # votes[((df.values[:,0:-2]) == 2)] += 2
        # votes[((df.values[:,0:-2]) == 6) | ((df.values[:,0:-2]) == 3)] += 1
        # np.array(votes.sum(axis=1)).max()
        mid_list = np.linspace(1, vals.shape[0], num= vals.shape[0])

        max_index, max_value = max(enumerate(vals.sum(axis=1)), key=operator.itemgetter(1))
        vals[max_index] = 0
        mid_list[max_index]=0
        max_index2, max_value2 = max(enumerate(vals.sum(axis=1)), key=operator.itemgetter(1))
        mid_list[max_index2]=0
        nrelationships = np.size(np.nonzero(votes[max_index,:]))

        # fam_list.append((zip(*df), max_index, max_value, nrelationships))
        # uz_df =
        fam_list.append((zip(*df), max_index, max_value, nrelationships))

        mlist = (f"MID{int(mid)}" for mid in mid_list if mid > 0)

        if nrelationships >= 3:
            fam_list2.append(
                [
                    df[1],
                    f"MID{str(1 + max_index)}",
                    max_value,
                    nrelationships,
                    f"MID{str(1 + max_index2)}",
                    max_value2,
                    np.size(np.nonzero(votes[max_index2, :])),
                    zip(*mlist),
                ]
            )

            tr_mids.append((df[1], mlist))

    with open('ttest.csv', "w") as ofile:
        writer = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

        for row in fam_list2:
            writer.writerow(row)
    for fl in fam_list2:
        print(fl)

    df_fams = pd.DataFrame(fam_list, columns=['rel', 'MID', 'val', 'nrel'])
    tr_list = []
    te_list = [
        glob.glob(dir_fid_root + tt[0] + "/" + tt[1] + "/*.jpg")
        for tt in fam_list2
    ]

    val_list = [
        glob.glob(dir_fid_root + tt[0] + "/" + tt[4] + "/*.jpg")
        for tt in fam_list2
    ]

    for tt in fam_list2:
        tr_list.extend(
            glob.glob(dir_fid_root + tt[0] + "/" + ttt + "/*.jpg")
            for ttt in list(tt[7:])
        )

    with open('test_no_labels.list', 'w') as f:
        for _list in te_list:
            if len(_list) == 0:
                continue
            fid = _list[0][69:74]
            # for _string in _list:
            for token in _list:
                f.write(f"{str(token)} " + fid + '\n')

    with open('val_no_labels.list', 'w') as f:
        for _list in val_list:
            if len(_list) == 0:
                continue
            fid = _list[0][69:74]
            # for _string in _list:
            for token in _list:
                f.write(f"{str(token)} " + fid + '\n')

    with open('train.list', 'w') as f:
        for _list in tr_list:
            if len(_list) == 0:
                continue
            fid = _list[0][69:74]
            # for _string in _list:
            for token in _list:
                f.write(f"{str(token)} " + fid + '\n')

                #     print(io.parent_dir(_list))
                # # f.seek(0)
                # f.write(str(_string) + '\n')

    # ofile  = open('test.;', "w")
class Pairs(object):
    def __init__(self, pair_list, kind=''):
        self.df_pairs = Pairs.list2table(pair_list)
        # self.df_pairs = pd.DataFrame({'p1': p1, 'p2': p2})
        self.type = kind
        self.npairs = len(self.df_pairs)

    @staticmethod
    def list2table(pair_list):
        p1 = [f'{pair.fid}/MID{pair.mids[0]}' for pair in pair_list]
        p2 = [f'{pair.fid}/MID{pair.mids[1]}' for pair in pair_list]

        return pd.DataFrame({'p1': p1, 'p2': p2})

    def write_pairs(self, f_path):
        """
        :param f_path: filepath (CSV file) to store all pairs
        :type f_path: str
        :return: None
        """
        self.df_pairs.to_csv(f_path, index=False)

        # def __str__(self):
        #     return "FID: {}\nMIDS: ({}, {})\tType: {}".format(self.fid, self.mids[0], self.mids[1], self.type)


class Pair(object):
    def __init__(self, mids, fid, kind=''):
        self.mids = mids
        self.fid = fid
        self.type = kind

    def __str__(self):
        return f"FID: {self.fid} ; MIDS: ({self.mids[0]}, {self.mids[1]}) ; Type: {self.type}"

    def __key(self):
        return self.mids[0], self.mids[1], self.fid, self.type

    # def __eq__(self, other):
    #     return self.fid == other.fid and self.mids[0] == other.mids[0] and self.mids[1] == other.mids[1]

    def __eq__(self, other):
        return self.__key() == other.__key()

    def __hash__(self):
        return hash(self.__key())

    def __lt__(self, other):
        return np.uint(self.fid[1::]) < np.uint(other.fid[1::])

