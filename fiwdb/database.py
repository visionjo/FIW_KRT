import glob

import pandas as pd
import numpy as np
from common.io import sys_home as dir_home

def get_unique_pairs(ids_in):
    """

    :param ids_in:
    :return:
    """
    ids = [(p1, p2) if p1 < p2 else (p2, p1) for p1, p2 in zip(list(ids_in[0]), list(ids_in[1]))]
    return list(set(ids))


def load_rid_lut(f_csv=dir_home() + "/Dropbox/Families_In_The_Wild/Database/FIW_RIDs.csv"):
    """

    :param f_csv:
    :return:
    """

    return pd.read_csv(f_csv, delimiter=',')


def load_pid_lut(f_csv=dir_home() + "/Dropbox/Families_In_The_Wild/Database/FIW_PIDs_new.csv"):
    """

    :param f_csv:
    :return:
    """
    return pd.read_csv(f_csv, delimiter='\t')


def load_fid_lut(f_csv=dir_home() + "/Dropbox/Families_In_The_Wild/Database/FIW_FIDs.csv"):
    """
    Load FIW_FIDs.csv-- FID- Surname LUT
    :param f_csv:
    :return:
    """
    return pd.read_csv(f_csv, delimiter='\t')


def load_fids(dirs_fid):
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

    return [pd.read_csv(d + f_csv) for d in dirs_fid]


def load_relationship_matrices(dirs_fid, f_csv='relationships.csv'):
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


def specify_gender(rel_mat, genders, gender):
    """
    :param rel_mat:
    :param genders: list of genders
    :param gender:  gender to search for {'Male' or Female}
    :type gender:   str
    :return:
    """
    ids_not = [j for j, s in enumerate(genders) if gender not in s]
    rel_mat[ids_not, :] = 0
    rel_mat[:, ids_not] = 0

    return rel_mat


class Pairs(object):
    def __init__(self, pair_list, kind=''):
        self.df_pairs = Pairs.list2table(pair_list)
        # self.df_pairs = pd.DataFrame({'p1': p1, 'p2': p2})
        self.type = kind
        self.npairs = len(self.df_pairs)

    @staticmethod
    def list2table(pair_list):
        p1 = ['{}/MID{}'.format(pair.fid, pair.mids[0]) for pair in pair_list]
        p2 = ['{}/MID{}'.format(pair.fid, pair.mids[1]) for pair in pair_list]

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
        return "FID: {} ; MIDS: ({}, {}) ; Type: {}".format(self.fid, self.mids[0], self.mids[1], self.type)

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
