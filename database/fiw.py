import pandas as pd
import glob
import numpy as np
import common.io as myio
import warnings as warn

def load_pids(f_csv = "/Users/josephrobinson/Dropbox/Families_In_The_Wild/Database/FIW_PIDs_new.csv"):
    """

    :param f_csv:
    :return:
    """
    return pd.read_csv(f_csv, delimiter='\t')

def load_fid_csv(f_csv = "/Users/josephrobinson/Dropbox/Families_In_The_Wild/Database/FIW_FIDs.csv"):
    """
    Load FIW_FIDs.csv-- FID- Surname LUT
    :param f_csv:
    :return:
    """
    return pd.read_csv(f_csv, delimiter='\t')





def load_families(dir_fids, f_rel_matrix='relationships.csv', f_mids='mids.csv'):
    """

    :param dir_fids: root folder containing FID/MID/ folders of DB.
    :return:
    """
    # family directories
    dirs_fid = glob.glob(dir_fids + '/F????/')
    fids = [d[-6:-1] for d in dirs_fid]

    fid_lut = load_fid_csv()
    # Load MID LUT for all FIDs.
    df_mids = load_mids(dirs_fid, f_csv=f_mids)

    print("{} families are being processed".format(len(df_mids)))

    # Load relationship matrices for all FIDs.
    relationship_matrices = load_relationship_matrices(dirs_fid, f_csv=f_rel_matrix)

    fams = []
    for i, mids in enumerate(df_mids):
        fid = fids[i]
        rel_matrix = relationship_matrices[i]
        nmember = mids['MID'].max()

        ids = list(fid_lut['FIDs']).index(fid)
        surname = str(np.array(fid_lut['surnames'])[ids])

        fams.append(Family(surname=surname, fid=fid, nmember=nmember, mid_lut=mids, relationship_matrix=rel_matrix))

    return fams


class Family(object):
    def __init__(self, surname='', fid=None, npids=0, nmember=0, mid_lut=None, relationship_matrix=None):
        self.surname = surname
        self.fid = fid
        self.npids = npids
        self.family_size = nmember
        self.relationship_matrix = relationship_matrix
        self.mid_lut = mid_lut


    def load_family(self):
        """

        :return:
        """
        pass


# def parse_sisters(dir_data = '/Users/josephrobinson//Dropbox/Families_In_The_Wild/Database/FIDs/',
#                    f_rel_matrix = 'relationships.csv', f_mids = 'mids.csv'):
#     """
#     Parse brother pairs by referencing member ID LUT and relationship matrix.
#
#     Siblings RID is 2 and brother, of course, are Males. Thus, these are the factors we use to identify pairs.
#
#     :param dir_data:        Directory containing folders of FIDs (e.g., F0001/, ..., F????/).
#     :param f_rel_matrix:
#     :param f_mids:
#     :return:
#     """
#
#     families = load_families(dir_data, f_rel_matrix=f_rel_matrix, f_mids=f_mids)
#     brothers = []
#     for i, fam in enumerate(families):
#         # ids = [i for i, s in enumerate(genders) if 'Male' in s]
#         rel_mat = np.array(fam.relationship_matrix)
#         genders = list(fam.mid_lut.Gender)
#         ids_not = [j for j, s in enumerate(genders) if 'Male' not in s]
#         rel_mat[ids_not, :] = 0
#         rel_mat[:, ids_not] = 0
#
#         brother_ids = np.where(rel_mat == 2)
#         npairs = len(brother_ids[1])
#         if npairs == 0:
#             warn.warn("No Brothers in " + str(fam))
#             continue
#         if npairs % 2 != 0:
#             warn.warn("Number of pairs should be even, but there are" + str(npairs))
#
#         for ids in enumerate(brother_ids):
#             print(ids)
#             indices = list(ids[1])
#             indices.sort()
#             brothers.append(Pairs(mids=indices, fid=fam.fid, type='brothers'))
#             del indices
#
#     return brothers

def load_fids(dirs_fid):
    """
    Function loads fid directories and labels.
    :param dirs_fid: root folder containing FID directories (i.e., F0001-F1000)
    :return: (list, list):  (fid filepaths and fid labels of these)
    """
    dirs = glob.glob(dirs_fid + 'F????/')
    fid_list = [d[-6:-1] for d in dirs]

    return dirs, fid_list


def load_mids(dirs_fid, f_csv='mids.csv'):
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
    df_relationships =[pd.read_csv(d + f_csv) for d in dirs_fid]

    for i in range(len(df_relationships)):
        # df_relationships = [content.ix[:, 1:len(content) + 1] for content in df_rel_contents]

        df_relationships[i].index = range(1, len(df_relationships[i]) + 1)
        df_relationships[i] = df_relationships[i].ix[:, 1:len(df_relationships[i]) + 1]

    return df_relationships


def parse_brothers(dir_data = '/Users/josephrobinson//Dropbox/Families_In_The_Wild/Database/FIDs/',
                   f_rel_matrix = 'relationships.csv', f_mids = 'mids.csv'):
    """
    Parse brother pairs by referencing member ID LUT and relationship matrix.

    Siblings RID is 2 and brother, of course, are Males. Thus, these are the factors we use to identify pairs.

    :param dir_data:        Directory containing folders of FIDs (e.g., F0001/, ..., F????/).
    :param f_rel_matrix:
    :param f_mids:
    :return:
    """

    # family directories
    dirs_fid, fid_list = load_fids(dir_data)

    nfids = len(fid_list)
    # Load MID LUT for all FIDs.
    df_mids = [pd.read_csv(d + f_mids) for d in dirs_fid]

    print("{} families are being processed".format(len(df_mids)))

    # Load relationship matrices for all FIDs.
    # df_relationships = [pd.read_csv(d + f_rel_matrix) for d in dirs_fid]
    #
    # for i in range(nfids):
    #     # df_relationships = [content.ix[:, 1:len(content) + 1] for content in df_rel_contents]
    #     df_relationships[i].index = range(1, len(df_relationships[i]) + 1)
    #     df_relationships[i] = df_relationships[i].ix[:, 1:len(df_relationships[i]) + 1]

    df_relationships = load_relationship_matrices(dirs_fid)
    brothers = []
    for i, fid in enumerate(fid_list):
        # ids = [i for i, s in enumerate(genders) if 'Male' in s]
        rel_mat = np.array(df_relationships[i])
        genders = list(df_mids[i].Gender)
        ids_not = [j for j, s in enumerate(genders) if 'Male' not in s]
        rel_mat[ids_not, :] = 0
        rel_mat[:, ids_not] = 0

        brother_ids = np.where(rel_mat == 2)
        npairs = len(brother_ids[1])
        if npairs == 0:
            print("No brothers in " + str(fid))
            continue
        if npairs % 2 != 0:
            warn.warn("Number of pairs should be even, but there are" + str(npairs))
        b_ids = [(b1, b2) if b1 < b2 else (b2, b1) for b1, b2 in zip(list(brother_ids[0]), list(brother_ids[1]))]
        b_ids = list(set(b_ids))
        for ids in enumerate(b_ids):
            print(ids)
            indices = list(np.array(ids[1]) + 1)
            brothers.append(Pairs(mids=indices, fid=fid, type='brothers'))
            del indices

    return brothers


def parse_sisters(dir_data = '/Users/josephrobinson//Dropbox/Families_In_The_Wild/Database/FIDs/',
                   f_rel_matrix = 'relationships.csv', f_mids = 'mids.csv'):
    """
    Parse sister pairs by referencing member ID LUT and relationship matrix.

    Siblings RID is 2 and sister, of course, are Females. Thus, these are the factors we use to identify pairs.

    :param dir_data:        Directory containing folders of FIDs (e.g., F0001/, ..., F????/).
    :param f_rel_matrix:
    :param f_mids:
    :return:
    """

    # family directories
    dirs_fid = glob.glob(dir_data + 'F????/')
    fids = [d[-6:-1] for d in dirs_fid]

    nfids = len(fids)
    # Load MID LUT for all FIDs.
    df_mids = [pd.read_csv(d + f_mids) for d in dirs_fid]

    print("{} families are being processed".format(len(df_mids)))

    # Load relationship matrices for all FIDs.
    df_relationships = [pd.read_csv(d + f_rel_matrix) for d in dirs_fid]

    for i in range(nfids):
        # df_relationships = [content.ix[:, 1:len(content) + 1] for content in df_rel_contents]
        df_relationships[i].index = range(1, len(df_relationships[i]) + 1)
        df_relationships[i] = df_relationships[i].ix[:, 1:len(df_relationships[i]) + 1]

    sisters = []
    for i, fid in enumerate(fids):
        # ids = [i for i, s in enumerate(genders) if 'Male' in s]
        rel_mat = np.array(df_relationships[i])
        genders = list(df_mids[i].Gender)
        ids_not = [j for j, s in enumerate(genders) if 'Female' not in s]
        rel_mat[ids_not, :] = 0
        rel_mat[:, ids_not] = 0

        sisters_ids = np.where(rel_mat == 2)
        npairs = len(sisters_ids[1])
        if npairs == 0:
            print("No sisters in " + str(fid))
            continue
        if npairs % 2 != 0:
            warn.warn("Number of pairs should be even, but there are" + str(npairs))
        b_ids = [(b1, b2) if b1 < b2 else (b2, b1) for b1, b2 in zip(list(sisters_ids[0]), list(sisters_ids[1]))]
        b_ids = list(set(b_ids))
        for ids in enumerate(b_ids):
            print(ids)
            indices = list(np.array(ids[1]) + 1)
            sisters.append(Pairs(mids=indices, fid=fid, type='brothers'))
            del indices

    return sisters


class Pairs(object):
    def __init__(self, mids, fid, type=''):
        self.mids = mids
        self.fid = fid
        self.type = type

    def __str__(self):
        return "FID: {}\nMIDS: ({}, {})\tType: {}".format(self.fid, self.mids[0], self.mids[1], self.type)

    def __key(self):
        return (self.mids[0], self.mids[1], self.fid, self.type)

    # def __eq__(self, other):
    #     return self.fid == other.fid and self.mids[0] == other.mids[0] and self.mids[1] == other.mids[1]

    def __eq__(self, other):
        return self.__key() == other.__key()

    def __hash__(self):
        return hash(self.__key())

    def __lt__(self, other):
        return np.uint(self.fid[1::]) < np.uint(other.fid[1::])


if __name__ == '__main__':
    bros = parse_brothers()
    print(len(bros))
    for index in range(0, 5):
        print(str(bros[index]))

    # bros = list(set(bros))
    # bros.sort()
    # print(len(bros))
    # for index in range(0, 15):
    #     print(str(bros[index]))


        # FID: F0001
        # MIDS: (3, 4)
        # Type: brothers
        # FID: F0007
        # MIDS: (1, 8)
        # Type: brothers
        # FID: F0008
        # MIDS: (1, 4)
        # Type: brothers
        # FID: F0008
        # MIDS: (8, 10)
        # Type: brothers
        # FID: F0009
        # MIDS: (1, 2)
        # Type: brothers

