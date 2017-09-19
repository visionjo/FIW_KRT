import pandas as pd
import glob
import numpy as np
import warnings as warn


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

    # family directories
    dirs_fid, fid_list = load_fids(dir_data)

    nfids = len(fid_list)
    print("{} families are being processed".format(nfids))
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
        npairs = len(sibling_ids[1])
        if npairs == 0:
            print("No brothers in " + str(fid))
            continue
        if npairs % 2 != 0:
            warn.warn("Number of pairs should be even, but there are" + str(npairs))
        sib_ids = [(b1, b2) if b1 < b2 else (b2, b1) for b1, b2 in zip(list(sibling_ids[0]), list(sibling_ids[1]))]
        sib_id = list(set(sib_ids))

        sibling_ids = list(set(sib_ids))
        for ids in sib_id:
            # remove if brother or sister pair
            if ('Male' in genders[ids[0]] and 'Male' in genders[ids[1]]) or \
                        ('Female' in genders[ids[0]] and 'Female' in genders[ids[1]]):
                    print("Removing", ids)
                    sibling_ids.remove(ids)

        for ids in enumerate(sibling_ids):
            print(ids)
            indices = list(np.array(ids[1]) + 1)
            siblings.append(Pair(mids=indices, fid=fid, kind='siblings'))
            del indices

    return siblings


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

def set_pairs(mylist, ids_in, kind, fid):
    ids = [(p1, p2) if p1 < p2 else (p2, p1) for p1, p2 in zip(list(ids_in[0]), list(ids_in[1]))]
    ids = list(set(ids))
    for i in enumerate(ids):
        print(i)
        indices = list(np.array(i[1]) + 1)
        mylist.append(Pair(mids=indices, fid=fid, kind='brothers'))
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
    dirs_fid, fid_list = load_fids(dir_data)

    nfids = len(fid_list)
    print("{} families are being processed".format(nfids))
    # Load MID LUT for all FIDs.
    df_mids = load_mids(dirs_fid, f_csv=f_mids)

    # Load relationship matrices for all FIDs.
    df_relationships = load_relationship_matrices(dirs_fid, f_csv=f_rel_matrix)
    brothers = []
    for i, fid in enumerate(fid_list):
        # ids = [i for i, s in enumerate(genders) if 'Male' in s]
        rel_mat = np.array(df_relationships[i])
        genders = list(df_mids[i].Gender)
        # zero out non-male subjects
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
    dirs_fid, fid_list = load_fids(dir_data)

    nfids = len(fid_list)
    print("{} families are being processed".format(nfids))
    # Load MID LUT for all FIDs.
    df_mids = load_mids(dirs_fid, f_csv=f_mids)

    # Load relationship matrices for all FIDs.
    df_relationships = load_relationship_matrices(dirs_fid, f_csv=f_rel_matrix)
    sisters = []
    for i, fid in enumerate(fid_list):
        # ids = [i for i, s in enumerate(genders) if 'Male' in s]
        rel_mat = np.array(df_relationships[i])
        genders = list(df_mids[i].Gender)


        ids_not = [j for j, s in enumerate(genders) if 'Female' not in s]
        rel_mat[ids_not, :] = 0
        rel_mat[:, ids_not] = 0

        sister_ids = np.where(rel_mat == 2)
        npairs = len(sister_ids[1])
        if npairs == 0:
            print("No sisters in " + str(fid))
            continue
        if npairs % 2 != 0:
            warn.warn("Number of pairs should be even, but there are" + str(npairs))
        s_ids = [(s1, s2) if s1 < s2 else (s2, s1) for s1, s2 in zip(list(sister_ids[0]), list(sister_ids[1]))]
        s_ids = list(set(s_ids))
        for ids in enumerate(s_ids):
            print(ids)
            indices = list(np.array(ids[1]) + 1)
            sisters.append(Pair(mids=indices, fid=fid, kind='sisters'))
            del indices

    return sisters


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
    dirs_fid, fid_list = load_fids(dir_data)

    nfids = len(fid_list)
    print("{} families are being processed".format(nfids))
    # Load MID LUT for all FIDs.
    df_mids = load_mids(dirs_fid, f_csv=f_mids)

    # Load relationship matrices for all FIDs.
    df_relationships = load_relationship_matrices(dirs_fid, f_csv=f_rel_matrix)
    fd = []
    fs = []
    md = []
    ms = []
    for i, fid in enumerate(fid_list):
        # ids = [i for i, s in enumerate(genders) if 'Male' in s]
        rel_mat = np.array(df_relationships[i])
        genders = list(df_mids[i].Gender)
        # ids_not = [j for j, s in enumerate(genders) if 'Female' not in s]
        # rel_mat[ids_not, :] = 0
        # rel_mat[:, ids_not] = 0

        # indices of matrix containing 4 or 1; that the matrix is inversed across the diagonal
        mat_ids = np.where(rel_mat == 1) and np.where(rel_mat.T == 1), np.where(rel_mat == 1) and np.where(rel_mat.T == 4)

        c_ids = np.where(rel_mat == 1)
        p_ids = np.where(rel_mat == 4)
        if len(c_ids[0]) != len(p_ids[0]):
            warn.warn("Number of children and parents are different.")

        npairs = len(c_ids[0])
        if npairs == 0:
            print("No parent-children pairs in " + str(fid))
            continue

        if npairs % 2 != 0:
            warn.warn("Number of pairs should be even, but there are" + str(npairs))

        ch_ids = [(p1, p2) if p1 < p2 else (p2, p1) for p1, p2 in zip(list(c_ids[0]), list(c_ids[1]))]
        par_ids = [(p1, p2) if p1 < p2 else (p2, p1) for p1, p2 in zip(list(p_ids[0]), list(p_ids[1]))]
        # ch_ids = list(set(ch_ids))
        for c, p in zip(ch_ids, par_ids):
            print(c,p)
            indices = list(np.array(c) + 1)
            if 'Male' in genders[c[0]] and 'Male' in genders[c[1]]:
                fs.append(Pair(mids=indices, fid=fid, kind='fs'))
            if 'Female' in genders[indices[0]] and 'Female' in genders[indices[1]]:
                fs.append(Pair(mids=indices, fid=fid, kind='fs'))
                    # or \
                    #     ('Female' in genders[id[0]] and 'Female' in genders[id[1]]):
                    # print("Removing", id)
                    # sibling_ids.remove(id)


            # sisters.append(Pair(mids=indices, fid=fid, kind='sisters'))
            del indices

    return fd, fs, md, ms


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

    def __str__(self):
        return "FID: {}\nMIDS: ({}, {})\tType: {}".format(self.fid, self.mids[0], self.mids[1], self.type)


class Pair(object):
    def __init__(self, mids, fid, kind=''):
        self.mids = mids
        self.fid = fid
        self.type = kind

    def __str__(self):
        return "FID: {} ; MIDS: ({}, {}) ; Type: {}".format(self.fid, self.mids[0], self.mids[1], self.type)

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
    out_bin = "/Users/josephrobinson/Dropbox/Families_In_The_Wild/Database/Pairs/"
    dir_fids = "/Users/josephrobinson/Dropbox/Families_In_The_Wild/Database/FIDs/"

    do_sibs = True
    do_save = False

    if do_sibs:
        print("Parsing Brothers")
        bros = parse_brothers(dir_data=dir_fids)
        print(len(bros))
        for index in range(0, 5):
            print(str(bros[index]))

        pair_set = Pairs(bros, kind='brothers')
        if do_save:
            pair_set.write_pairs(out_bin + "bb1.csv")

        del bros, pair_set
    if False:
        print("Parsing Sisters")
        sis = parse_sisters(dir_data=dir_fids)
        print(len(sis))
        for index in range(0, 5):
            print(str(sis[index]))

        pair_set = Pairs(sis, kind='sisters')

        if do_save:
            pair_set.write_pairs(out_bin + "ss1.csv")

        del sis, pair_set

        print("Parsing Siblings")
        sibs = parse_siblings(dir_data=dir_fids)
        print(len(sibs))
        for index in range(0, 5):
            print(str(sibs[index]))

        pair_set = Pairs(sibs, kind='siblings')

        if do_save:
            pair_set.write_pairs(out_bin + "sibs1.csv")

        del sibs, pair_set

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
