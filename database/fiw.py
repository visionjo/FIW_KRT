import pandas as pd
import glob
import numpy as np
import warnings as warn
import itertools

import common.io as io

do_debug = False


def load_rid_lut(f_csv="/Users/josephrobinson/Dropbox/Families_In_The_Wild/Database/FIW_RIDs.csv"):
    """

    :param f_csv:
    :return:
    """

    return pd.read_csv(f_csv, delimiter=',')


def load_pids(f_csv="/Users/josephrobinson/Dropbox/Families_In_The_Wild/Database/FIW_PIDs_new.csv"):
    """

    :param f_csv:
    :return:
    """
    return pd.read_csv(f_csv, delimiter='\t')


def load_fid_csv(f_csv="/Users/josephrobinson/Dropbox/Families_In_The_Wild/Database/FIW_FIDs.csv"):
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


def load_fid_lut(dirs_fid):
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
    df_relationships = [pd.read_csv(d + f_csv) for d in dirs_fid]

    for i in range(len(df_relationships)):
        # df_relationships = [content.ix[:, 1:len(content) + 1] for content in df_rel_contents]

        df_relationships[i].index = range(1, len(df_relationships[i]) + 1)
        df_relationships[i] = df_relationships[i].ix[:, 1:len(df_relationships[i]) + 1]

    return df_relationships


def parse_siblings(dir_data='/Users/josephrobinson//Dropbox/Families_In_The_Wild/Database/FIDs/',
                   f_rel_matrix='relationships.csv', f_mids='mids.csv'):
    """
    Parse brother pairs by referencing member ID LUT and relationship matrix.

    Siblings RID is 2 and brother, of course, are Males. Thus, these are the factors we use to identify pairs.

    :param dir_data:        Directory containing folders of FIDs (e.g., F0001/, ..., F????/).
    :param f_rel_matrix:
    :param f_mids:
    :return:
    """
    kind = 'siblings'
    # family directories
    dirs_fid, fid_list = load_fid_lut(dir_data)

    print("{} families are being processed".format(len(fid_list)))
    # Load MID LUT for all FIDs.
    df_mids = load_mids(dirs_fid, f_csv=f_mids)

    # Load relationship matrices for all FIDs.
    df_relationships = load_relationship_matrices(dirs_fid, f_csv=f_rel_matrix)
    siblings = []
    for i, fid in enumerate(fid_list):
        # ids = [i for i, s in enumerate(genders) if 'Male' in s]
        rel_mat = np.array(df_relationships[i])
        genders = list(df_mids[i].Gender)
        # ids_not = [j for j, s in enumerate(genders) if 'Male' not in s]
        # rel_mat[ids_not, :] = 0
        # rel_mat[:, ids_not] = 0

        sibling_ids = np.where(rel_mat == 2)

        if not check_npairs(len(sibling_ids[1]), kind, fid):
            continue

        sib_ids = get_unique_pairs(sibling_ids)
        # sib_ids = [(b1, b2) if b1 < b2 else (b2, b1) for b1, b2 in zip(list(sibling_ids[0]), list(sibling_ids[1]))]
        # sib_id = list(set(sib_ids))

        sibling_ids = list(set(sib_ids))

        for ids in sib_ids:
            # remove if brother or sister pair
            if ('Male' in genders[ids[0]] and 'Male' in genders[ids[1]]) or \
                    ('Female' in genders[ids[0]] and 'Female' in genders[ids[1]]):
                print("Removing", ids)
                sibling_ids.remove(ids)

        for ids in enumerate(sibling_ids):
            print(ids)
            indices = list(np.array(ids[1]) + 1)
            siblings.append(Pair(mids=indices, fid=fid, kind=kind))
            del indices

    return siblings


def get_face_pairs(dirs_fid, df_pairs):
    """
    Generate table of all face combinations up for each pair
    :param dirs_fid:
    :param pairs:
    :return:
    """
    all_pairs = []
    for index, row in df_pairs.iterrows():
        paths1 = glob.glob(dirs_fid + "/" + row['p1'] + "/*.jpg")
        paths2 = glob.glob(dirs_fid + "/" + row['p2'] + "/*.jpg")

        faces1 = [p.replace(dir_fids + "/", "") for p in paths1]
        faces2 = [p.replace(dir_fids + "/", "") for p in paths2]

        # print(faces1, faces2)
        for x in faces1:
            for y in faces2:
                all_pairs.append([x, y])
                # [[x, y] for x, y in zip(faces1, faces2)]

    arr_pairs = np.array(all_pairs)

    print('No. Face Pairs is {}.'.format(arr_pairs.shape[0]))
    return pd.DataFrame({df_pairs.columns[0]: arr_pairs[:, 0], df_pairs.columns[1]: arr_pairs[:, 1]})


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


def check_npairs(npairs, ktype, fid):
    """
    Check tht pairs follow expected
    :param npairs:
    :return:    True if both test passes
    """
    if npairs == 0:
        print("No " + ktype + " in " + str(fid))
        return False
    if npairs % 2 != 0:
        warn.warn("Number of pairs should be even, but there are" + str(npairs))
        return False

    return True


def get_unique_pairs(ids_in):
    """

    :param ids_in:
    :return:
    """
    ids = [(p1, p2) if p1 < p2 else (p2, p1) for p1, p2 in zip(list(ids_in[0]), list(ids_in[1]))]
    return list(set(ids))


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


def parse_brothers(dir_data='/Users/josephrobinson//Dropbox/Families_In_The_Wild/Database/FIDs/',
                   f_rel_matrix='relationships.csv', f_mids='mids.csv'):
    """
    Parse brother pairs by referencing member ID LUT and relationship matrix.

    Siblings RID is 2 and brother, of course, are Males. Thus, these are the factors we use to identify pairs.

    :param dir_data:        Directory containing folders of FIDs (e.g., F0001/, ..., F????/).
    :param f_rel_matrix:
    :param f_mids:
    :return:
    """
    kind = 'brothers'
    # family directories
    dirs_fid, fid_list = load_fid_lut(dir_data)

    print("{} families are being processed".format(len(fid_list)))
    # Load MID LUT for all FIDs.
    df_mids = load_mids(dirs_fid, f_csv=f_mids)

    # Load relationship matrices for all FIDs.
    df_relationships = load_relationship_matrices(dirs_fid, f_csv=f_rel_matrix)
    brothers = []
    for i, fid in enumerate(fid_list):
        # ids = [i for i, s in enumerate(genders) if 'Male' in s]
        rel_mat = np.array(df_relationships[i])
        genders = list(df_mids[i].Gender)
        # zero out female subjects
        rel_mat = specify_gender(rel_mat, genders, 'Male')

        brother_ids = np.where(rel_mat == 2)

        if not check_npairs(len(brother_ids[1]), kind, fid):
            continue
        # add to list of brothers
        brothers = set_pairs(brothers, brother_ids, kind, fid)

    return brothers


def parse_sisters(dir_data='/Users/josephrobinson//Dropbox/Families_In_The_Wild/Database/FIDs/',
                  f_rel_matrix='relationships.csv', f_mids='mids.csv'):
    """
    Parse sister pairs by referencing member ID LUT and relationship matrix.

    Siblings RID is 2 and sister, of course, are Females. Thus, these are the factors we use to identify pairs.

    :param dir_data:        Directory containing folders of FIDs (e.g., F0001/, ..., F????/).
    :param f_rel_matrix:
    :param f_mids:
    :return:
    """

    # family directories
    kind = 'sisters'
    dirs_fid, fid_list = load_fid_lut(dir_data)

    print("{} families are being processed".format(len(fid_list)))
    # Load MID LUT for all FIDs.
    df_mids = load_mids(dirs_fid, f_csv=f_mids)

    # Load relationship matrices for all FIDs.
    df_relationships = load_relationship_matrices(dirs_fid, f_csv=f_rel_matrix)
    sisters = []
    for i, fid in enumerate(fid_list):
        # ids = [i for i, s in enumerate(genders) if 'Male' in s]
        rel_mat = np.array(df_relationships[i])
        genders = list(df_mids[i].Gender)

        # zero out female subjects
        rel_mat = specify_gender(rel_mat, genders, 'Female')

        sister_ids = np.where(rel_mat == 2)

        if not check_npairs(len(sister_ids[1]), kind, fid):
            continue

        # add to list of brothers
        sisters = set_pairs(sisters, sister_ids, kind, fid)

    return sisters


def tri_subjects(dir_data='/Users/josephrobinson//Dropbox/Families_In_The_Wild/Database/FIDs/',
                 f_rel_matrix='relationships.csv', f_mids='mids.csv'):
    """
    Parse sister pairs by referencing member ID LUT and relationship matrix.

    Siblings RID is 2 and sister, of course, are Females. Thus, these are the factors we use to identify pairs.

    :param dir_data:        Directory containing folders of FIDs (e.g., F0001/, ..., F????/).
    :param f_rel_matrix:
    :param f_mids:
    :return:
    """

    # family directories
    dirs_fid, fid_list = load_fid_lut(dir_data)

    print("{} families are being processed".format(len(fid_list)))
    # Load MID LUT for all FIDs.
    df_mids = load_mids(dirs_fid, f_csv=f_mids)

    # Load relationship matrices for all FIDs.
    df_relationships = load_relationship_matrices(dirs_fid, f_csv=f_rel_matrix)
    fms = []
    fmd = []
    kind = 'parents-child'
    for i, fid in enumerate(fid_list):
        # ids = [i for i, s in enumerate(genders) if 'Male' in s]
        rel_mat = np.array(df_relationships[i])
        genders = list(df_mids[i].Gender)

        # indices of matrix containing 4 or 1; that the matrix is inversed across the diagonal
        # mat_ids = np.where(rel_mat == 1) and np.where(rel_mat.T == 1), np.where(rel_mat == 1) and np.where(
        #     rel_mat.T == 4)

        c_ids = np.where(rel_mat == 1)
        p_ids = np.where(rel_mat == 4)
        ch_ids = [[p1, p2] for p1, p2 in zip(list(c_ids[0]), list(c_ids[1]))]
        par_ids = [[p1, p2] for p1, p2 in zip(list(p_ids[0]), list(p_ids[1]))]

        # check that children as 2 parents
        rows = np.array(ch_ids)[:, 1]
        cols = np.array(ch_ids)[:, 0]
        unique, counts = np.unique(rows, return_counts=True)
        count_check = counts != 2
        # ids = np.where(count_check == False)[0]
        ids = unique
        if len(ids) == 0:
            print("Two parents are not present for child")
            continue

        # cc_ids = cols[np.where(rows == ids[0])]

        p_cols = np.array(par_ids)[:, 0]

        for cc in ids:
            pars = rows[np.where(cols == cc)]
            if len(pars) != 2:
                continue
            try:
                p_genders = [genders[pars[0]], genders[pars[1]]]

            except IndexError:
                print()
            if "Male" in p_genders[0] and "Female" in p_genders[1]:
                pars_ids = (pars[0] + 1, pars[1] + 1)
            elif "Male" in p_genders[1] and "Female" in p_genders[0]:
                pars_ids = pars[1] + 1, pars[0] + 1
            else:
                warn.warn("Parents are of same gender for ", fid)
                continue

            cmid = "FID/{}/MIDS{}".format(fid, cc + 1)
            fmid = "FID/{}/MIDS{}".format(fid, pars_ids[0])
            mmid = "FID/{}/MIDS{}".format(fid, pars_ids[1])
            if "Male" in genders[cc]:
                fms.append((fmid, mmid, cmid))
            else:
                fmd.append((fmid, mmid, cmid))


                # # pp_ids = cols[np.where(rows == ids[1])]
                # p_ids = np.where(rel_mat == 4)
                #
                # if len(c_ids[0]) != len(p_ids[0]):
                #     warn.warn("Number of children and parents are different.")
                #
                # if not check_npairs(len(c_ids[0]), kind, fid):
                #     continue


                # # ch_ids = list(set(ch_ids))
                # for p in par_ids:
                #     print(p)
                #     p_mid = list(np.array(p) + 1)[0]
                #     c_mid = list(np.array(p) + 1)[1]
                #
                #     p_gender = genders[p_mid - 1]
                #     c_gender = genders[c_mid - 1]
                #     if 'Male' in p_gender:
                #         # fathers
                #         if 'Male' in c_gender:
                #             # son
                #             fs.append(Pair(mids=(p_mid, c_mid), fid=fid, kind='fs'))
                #         else:
                #             # daughter
                #             fd.append(Pair(mids=(p_mid, c_mid), fid=fid, kind='fd'))
                #     else:
                #         # mothers
                #         if 'Male' in c_gender:
                #             ms.append(Pair(mids=(p_mid, c_mid), fid=fid, kind='ms'))
                #         else:
                #             md.append(Pair(mids=(p_mid, c_mid), fid=fid, kind='md'))

    return fmd, fms


def parse_parents(dir_data='/Users/josephrobinson//Dropbox/Families_In_The_Wild/Database/FIDs/',
                  f_rel_matrix='relationships.csv', f_mids='mids.csv'):
    """
    Parse sister pairs by referencing member ID LUT and relationship matrix.

    Siblings RID is 2 and sister, of course, are Females. Thus, these are the factors we use to identify pairs.

    :param dir_data:        Directory containing folders of FIDs (e.g., F0001/, ..., F????/).
    :param f_rel_matrix:
    :param f_mids:
    :return:
    """

    # family directories
    dirs_fid, fid_list = load_fid_lut(dir_data)

    print("{} families are being processed".format(len(fid_list)))
    # Load MID LUT for all FIDs.
    df_mids = load_mids(dirs_fid, f_csv=f_mids)

    # Load relationship matrices for all FIDs.
    df_relationships = load_relationship_matrices(dirs_fid, f_csv=f_rel_matrix)
    fd = []
    fs = []
    md = []
    ms = []
    kind = 'parent-child'
    for i, fid in enumerate(fid_list):
        # ids = [i for i, s in enumerate(genders) if 'Male' in s]
        rel_mat = np.array(df_relationships[i])
        genders = list(df_mids[i].Gender)

        # ids_not = [j for j, s in enumerate(genders) if 'Female' not in s]
        # rel_mat[ids_not, :] = 0
        # rel_mat[:, ids_not] = 0

        # indices of matrix containing 4 or 1; that the matrix is inversed across the diagonal
        # mat_ids = np.where(rel_mat == 1) and np.where(rel_mat.T == 1), np.where(rel_mat == 1) and np.where(
        #     rel_mat.T == 4)

        c_ids = np.where(rel_mat == 1)
        p_ids = np.where(rel_mat == 4)
        if len(c_ids[0]) != len(p_ids[0]):
            warn.warn("Number of children and parents are different.")

        if not check_npairs(len(c_ids[0]), kind, fid):
            continue
        ch_ids = [(p1, p2) for p1, p2 in zip(list(c_ids[0]), list(c_ids[1]))]
        par_ids = [(p1, p2) for p1, p2 in zip(list(p_ids[0]), list(p_ids[1]))]

        # ch_ids = list(set(ch_ids))
        for p in par_ids:
            print(p)
            p_mid = list(np.array(p) + 1)[0]
            c_mid = list(np.array(p) + 1)[1]

            p_gender = genders[p_mid - 1]
            c_gender = genders[c_mid - 1]
            if 'Male' in p_gender:
                # fathers
                if 'Male' in c_gender:
                    # son
                    fs.append(Pair(mids=(p_mid, c_mid), fid=fid, kind='fs'))
                else:
                    # daughter
                    fd.append(Pair(mids=(p_mid, c_mid), fid=fid, kind='fd'))
            else:
                # mothers
                if 'Male' in c_gender:
                    ms.append(Pair(mids=(p_mid, c_mid), fid=fid, kind='ms'))
                else:
                    md.append(Pair(mids=(p_mid, c_mid), fid=fid, kind='md'))

    return fd, fs, md, ms


def parse_grandparents(dir_data='/Users/josephrobinson//Dropbox/Families_In_The_Wild/Database/FIDs/',
                       f_rel_matrix='relationships.csv', f_mids='mids.csv'):
    """
    Parse sister pairs by referencing member ID LUT and relationship matrix.

    Siblings RID is 2 and sister, of course, are Females. Thus, these are the factors we use to identify pairs.

    :param dir_data:        Directory containing folders of FIDs (e.g., F0001/, ..., F????/).
    :param f_rel_matrix:
    :param f_mids:
    :return:
    """

    # family directories
    dirs_fid, fid_list = load_fid_lut(dir_data)

    print("{} families are being processed".format(len(fid_list)))
    # Load MID LUT for all FIDs.
    df_mids = load_mids(dirs_fid, f_csv=f_mids)

    # Load relationship matrices for all FIDs.
    df_relationships = load_relationship_matrices(dirs_fid, f_csv=f_rel_matrix)
    gfgd = []
    gfgs = []
    gmgd = []
    gmgs = []
    kind = 'parent-child'
    for i, fid in enumerate(fid_list):
        # ids = [i for i, s in enumerate(genders) if 'Male' in s]
        rel_mat = np.array(df_relationships[i])
        genders = list(df_mids[i].Gender)

        # ids_not = [j for j, s in enumerate(genders) if 'Female' not in s]
        # rel_mat[ids_not, :] = 0
        # rel_mat[:, ids_not] = 0

        # indices of matrix containing 4 or 1; that the matrix is inversed across the diagonal
        # mat_ids = np.where(rel_mat == 1) and np.where(rel_mat.T == 1), np.where(rel_mat == 1) and np.where(
        #     rel_mat.T == 4)

        c_ids = np.where(rel_mat == 3)
        p_ids = np.where(rel_mat == 6)
        if len(c_ids[0]) != len(p_ids[0]):
            warn.warn("Number of children and parents are different.")

        if not check_npairs(len(c_ids[0]), kind, fid):
            continue
        # ch_ids = [(p1, p2) for p1, p2 in zip(list(c_ids[0]), list(c_ids[1]))]
        par_ids = [(p1, p2) for p1, p2 in zip(list(p_ids[0]), list(p_ids[1]))]

        # ch_ids = list(set(ch_ids))
        for p in par_ids:
            print(p)
            p_mid = list(np.array(p) + 1)[0]
            c_mid = list(np.array(p) + 1)[1]

            p_gender = genders[p_mid - 1]
            c_gender = genders[c_mid - 1]
            if 'Male' in p_gender:
                # fathers
                if 'Male' in c_gender:
                    # son
                    gfgs.append(Pair(mids=(p_mid, c_mid), fid=fid, kind='fs'))
                else:
                    # daughter
                    gfgd.append(Pair(mids=(p_mid, c_mid), fid=fid, kind='fd'))
            else:
                # mothers
                if 'Male' in c_gender:
                    gmgs.append(Pair(mids=(p_mid, c_mid), fid=fid, kind='ms'))
                else:
                    gmgd.append(Pair(mids=(p_mid, c_mid), fid=fid, kind='md'))

    return gfgd, gfgs, gmgd, gmgs


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


def compare_lists(list1, list2):
    """
    Compares sizes and contents of 2 lists.
    :param list1:
    :param list2:
    :return: True, True: if both sizes and contents are equal; True, False: size same, content differs; etc.
    """

    same_size = True
    same_contents = True

    if len(list1) != len(list2):
        same_size = False

    if list1 != list2:
        same_contents = False

    return same_size, same_contents


def check_rel_matrix(rel_matrix, fid=''):
    """

    :param rel_matrix:
    :return:    True if matrix passes all tests
    """
    passes = True
    messages = []
    # check diagonal is all zeros
    if any(rel_matrix.diagonal() != 0):
        messages.append("Non-zero elements found in diagonal of relationship matrix ({})".format(fid))
        warn.warn(messages[len(messages) - 1])
        passes = False

    rids = load_rid_lut()

    pair_types = [(rids.RID[1], rids.RID[1]),   # siblings
                  (rids.RID[0], rids.RID[3]),   # parent-child
                  (rids.RID[2], rids.RID[5]),   # grandparent-grandchild
                  (rids.RID[4], rids.RID[4]),   # spouses
                  (rids.RID[6], rids.RID[7])    # great-grandparent-great-grandchild
                  ]

    # check that matrix elements of upper/ lower triangle correspond (e.g., 4 at (4,3) means 1 at (3, 4)
    # do this for each type
    for int_pair in pair_types:
        n_mismatches = (np.where(rel_matrix == int_pair[0], 1, 0) - np.where(rel_matrix == int_pair[1], 1, 0).T).sum()

        if n_mismatches > 0:
            messages.append("Inconsistent labels ({}) in relationship matrix ({}) for RIDs {}".format(n_mismatches,
                                                                                                      fid,
                                                                                                      int_pair))
            warn.warn(messages[len(messages) - 1])
            passes = False

    return passes, messages


def perpare_fids(dir_fid="/Users/josephrobinson/Dropbox/Families_In_The_Wild/Database/Ann/FW_FIDs/",
                 dirs_out="/Users/josephrobinson/Dropbox/Families_In_The_Wild/Database/FIDs/"):
    """
    Parses FID CSV files and places in DB. Additionally, checks are made for inconsistency in labels.
    :param dir_fid:
    :param dirs_out:
    :return:
    """

    # load fids (CSV files)
    fid_files = glob.glob(dir_fid + "F????.csv")
    fid_dicts = {io.file_base(f): pd.read_csv(f) for f in fid_files}

    # dfs_fids, fids = [(pd.read_csv(f), io.file_base(f)) for f in fid_files]

    for fid in fid_dicts:
        # for each fid (i.e., iterate keys of dictionary)
        print(fid)
        df_rel_mat = fid_dicts[fid]
        col_gender = df_rel_mat.Gender

        # get MIDs
        ids = [i for i, c in enumerate(col_gender) if '-1' not in c]
        tmp = list(range(0, len(ids)))

        same_size, same_contents = compare_lists(tmp, ids)

        if not (same_size and same_contents):
            warn.warn("MIDs and row indices of relationship table differ")
        mids = list(np.array(ids) + 1)
        cols = df_rel_mat.columns[mids]
        df_rel_mat.loc[ids][cols]

        rel_mat = np.array(df_rel_mat.loc[ids][cols])
        success, messages = check_rel_matrix(rel_mat, fid=fid)
        if not success:
            [print(p) for p in success]
            continue

        genders = [g.lower() for g in list(col_gender[ids])]
        names = list(df_rel_mat.Name[ids])

        df_fam = pd.DataFrame({'MID': mids})
        df_fam = df_fam.join(df_rel_mat.loc[ids][cols])
        df_fam = df_fam.join(pd.DataFrame({'Gender': genders, 'Name': names}))
        df_fam.to_csv(dirs_out + "/" + fid + "/mid.csv",index=False)


if __name__ == '__main__':
    out_bin = "/Users/josephrobinson/Dropbox/Families_In_The_Wild/Database/Pairs/"
    dir_fids = "/Users/josephrobinson/Dropbox/Families_In_The_Wild/Database/FIDs/"
    dir_fid = "/Users/josephrobinson/Dropbox/Families_In_The_Wild/Database/Ann/FW_FIDs/"
    do_sibs = False
    do_save = True

    if do_sibs:
        print("Parsing Brothers")
        bros = parse_brothers(dir_data=dir_fids)
        print(len(bros))
        for index in range(0, 5):
            print(str(bros[index]))

        pair_set = Pairs(bros, kind='brothers')

        df_all_faces = get_face_pairs(dir_fids, pair_set.df_pairs)

        if do_save:
            pair_set.write_pairs(out_bin + "bb-pairs.csv")
            df_all_faces.to_csv(out_bin + 'bb-faces.csv', index=False)

        del bros, pair_set, df_all_faces

        print("Parsing Sisters")
        sis = parse_sisters(dir_data=dir_fids)
        print(len(sis))
        for index in range(0, 5):
            print(str(sis[index]))

        pair_set = Pairs(sis, kind='sisters')

        df_all_faces = get_face_pairs(dir_fids, pair_set.df_pairs)

        if do_save:
            pair_set.write_pairs(out_bin + "ss-pairs.csv")
            df_all_faces.to_csv(out_bin + 'ss-faces.csv', index=False)

        del sis, pair_set, df_all_faces

        print("Parsing Siblings")
        sibs = parse_siblings(dir_data=dir_fids)
        print(len(sibs))
        for index in range(0, 5):
            print(str(sibs[index]))

        pair_set = Pairs(sibs, kind='siblings')

        df_all_faces = get_face_pairs(dir_fids, pair_set.df_pairs)
        if do_save:
            pair_set.write_pairs(out_bin + "sibs-pairs.csv")
            df_all_faces.to_csv(out_bin + 'sibs-faces.csv', index=False)

        del sibs, pair_set, df_all_faces

        gfgd, gfgs, gmgd, gmgs = parse_grandparents(dir_data=dir_fids)

    if False:
        print("Parsing Parents")
        fd, fs, md, ms = parse_parents(dir_data=dir_fids)

        fd_set = Pairs(fd, kind='fd')
        df_all_faces = get_face_pairs(dir_fids, fd_set.df_pairs)
        fd_set.write_pairs(out_bin + "fd-pairs.csv")
        df_all_faces.to_csv(out_bin + 'fd-faces.csv', index=False)
        print(len(df_all_faces))
        del df_all_faces

        fs_set = Pairs(fs, kind='fs')
        df_all_faces = get_face_pairs(dir_fids, fs_set.df_pairs)
        fs_set.write_pairs(out_bin + "fs-pairs.csv")
        df_all_faces.to_csv(out_bin + 'fs-faces.csv', index=False)
        print(len(df_all_faces))
        del df_all_faces

        md_set = Pairs(md, kind='md')
        df_all_faces = get_face_pairs(dir_fids, md_set.df_pairs)
        md_set.write_pairs(out_bin + "md-pairs.csv")
        df_all_faces.to_csv(out_bin + 'md-faces.csv', index=False)
        print(len(df_all_faces))
        del df_all_faces

        ms_set = Pairs(ms, kind='ms')
        df_all_faces = get_face_pairs(dir_fids, ms_set.df_pairs)
        ms_set.write_pairs(out_bin + "ms-pairs.csv")
        df_all_faces.to_csv(out_bin + 'ms-faces.csv', index=False)
        print(len(df_all_faces))
        del df_all_faces
    if False:
        fmd, fms = tri_subjects(dir_data=dir_fids)
        print(len(fmd))
        for index in range(0, 5):
            print(str(fmd[index]))
        print(len(fms))
        for index in range(0, 5):
            print(str(fms[index]))
    perpare_fids(dir_fid=dir_fid, dirs_out=dir_fids)
    print()
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


    # 655
    # FID: F0003 ; MIDS: (1, 4) ; Type: sisters
    # FID: F0004 ; MIDS: (4, 5) ; Type: sisters
    # FID: F0006 ; MIDS: (3, 4) ; Type: sisters
    # FID: F0007 ; MIDS: (5, 6) ; Type: sisters
    # FID: F0007 ; MIDS: (4, 5) ; Type: sisters


    # 1217
    # FID: F0004 ; MIDS: (2, 4) ; Type: siblings
    # FID: F0004 ; MIDS: (2, 5) ; Type: siblings
    # FID: F0007 ; MIDS: (8, 9) ; Type: siblings
    # FID: F0007 ; MIDS: (3, 4) ; Type: siblings
    # FID: F0007 ; MIDS: (3, 6) ; Type: siblings
    #
    # Process finished with exit code 0

    # 698 - no. unique pairs
    # 49998 - total brothers
    # 655 - no. unique pairs
    # 19349
    # 1217 - no. unique pairs
    # 36023

    # fd 910
    #     33352
    # fs 972
    #     43993
    # md 939
    # ms 957
