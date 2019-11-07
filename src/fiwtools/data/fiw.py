from __future__ import print_function
import glob
import warnings as warn
import numpy as np
import pandas as pd
import csv

from src.fiwtools.utils import io
import src.fiwtools.fiwdb.database as db
import src.fiwtools.fiwdb.helpers as helpers
from src.fiwtools.fiwdb.database import load_fids

from collections import defaultdict
import src.fiwtools.utils.log as log
from src.fiwtools.utils.io import sys_home as dir_home
from src.fiwtools.data import fiw

logger = log.setup_custom_logger(__name__)
logger.debug('Parse FIW')

# import logging

#
# logging.basicConfig(filename="fiw_1.log", level=logging.ERROR, format="%(asctime)s:%(levelname)s:%(message)s")
# log = logging.getLogger(__name__)

do_debug = False


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



def write_list_tri_pairs(fout, l_tuples):
    with open(fout, 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['father', 'mother', 'child'])
        for row in l_tuples:
            csv_out.writerow(row)


class Member(object):
    def __init__(self, mid, gender, father, mother, brothers, sisters):
        pass


class Family(object):
    def __init__(self, surname='', fid=None, npids=0, nmember=0, mid_lut=None, relationship_matrix=None):
        self.surname = surname
        self.fid = fid
        self.npids = npids
        self.family_size = nmember
        self.relationship_matrix = relationship_matrix
        self.mid_lut = mid_lut
        self.members = {}  # {MID: Member}

    def load_family(self):
        """

        :return:
        """
        pass

def load_families(dir_fids, f_mids='mid.csv'):
    """

    :param dir_fids: root folder containing FID/MID/ folders of DB.
    :return:
    """
    # family directories
    dirs_fid = glob.glob(dir_fids + '/F????/')
    fids = [d[-6:-1] for d in dirs_fid]

    fid_lut = db.load_fids()
    # Load MID LUT for all FIDs.
    df_mids = db.load_mids(dirs_fid, f_csv=f_mids)

    print("{} families are being processed".format(len(df_mids)))

    # Load relationship matrices for all FIDs.
    relationship_matrices = db.load_relationship_matrices(dirs_fid, f_csv=f_mids)

    fams = []
    for i, mids in enumerate(df_mids):
        fid = fids[i]
        rel_matrix = relationship_matrices[i]
        nmember = mids['MID'].max()

        ids = list(fid_lut['FIDs']).index(fid)
        surname = str(np.array(fid_lut['surnames'])[ids])

        fams.append(Family(surname=surname, fid=fid, nmember=nmember, mid_lut=mids, relationship_matrix=rel_matrix))

    return fams


def parse_siblings(dir_data='/Users/josephrobinson//Dropbox/Families_In_The_Wild/Database/FIDs/', f_mids='mid.csv'):
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
    dirs_fid, fid_list = load_fids(dir_data)

    print("{} families are being processed".format(len(fid_list)))
    # Load MID LUT for all FIDs.
    df_mids = db.load_mids(dirs_fid, f_csv=f_mids)

    # Load relationship matrices for all FIDs.
    # df_relationships = load_relationship_matrices(dirs_fid, f_csv=f_rel_matrix)
    siblings = []
    for i, fid in enumerate(fid_list):
        # ids = [i for i, s in enumerate(genders) if 'Male' in s]
        rel_mat = db.parse_relationship_matrices(df_mids[i])
        # rel_mat = np.array(df_relationships[i])
        genders = list(df_mids[i].Gender)
        # ids_not = [j for j, s in enumerate(genders) if 'Male' not in s]
        # rel_mat[ids_not, :] = 0
        # rel_mat[:, ids_not] = 0

        sibling_ids = np.where(rel_mat == 2)

        if not helpers.check_npairs(len(sibling_ids[1]), kind, fid):
            continue

        sib_ids = db.get_unique_pairs(sibling_ids)
        # sib_ids = [(b1, b2) if b1 < b2 else (b2, b1) for b1, b2 in zip(list(sibling_ids[0]), list(sibling_ids[1]))]
        # sib_id = list(set(sib_ids))

        sibling_ids = list(set(sib_ids))

        for ids in sib_ids:
            # remove if brother or sister pair
            if ('m' in genders[ids[0]] and 'f' in genders[ids[1]]) or \
                    ('f' in genders[ids[0]] and 'm' in genders[ids[1]]):
                print("Removing", ids)
                sibling_ids.remove(ids)

        for ids in enumerate(sibling_ids):
            print(ids)
            indices = list(np.array(ids[1]) + 1)
            siblings.append(db.Pair(mids=indices, fid=fid, kind=kind))
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

        faces1 = [p.replace(dirs_fid + "/", "").replace(dirs_fid, "") for p in paths1]
        faces2 = [p.replace(dirs_fid + "/", "").replace(dirs_fid, "") for p in paths2]

        # print(faces1, faces2)
        for x in faces1:
            for y in faces2:
                all_pairs.append([x, y])
                # [[x, y] for x, y in zip(faces1, faces2)]

    arr_pairs = np.array(all_pairs)

    print('No. Face Pairs is {}.'.format(arr_pairs.shape[0]))
    return pd.DataFrame({df_pairs.columns[0]: arr_pairs[:, 0], df_pairs.columns[1]: arr_pairs[:, 1]})


def parse_brothers(dir_data='/Users/josephrobinson//Dropbox/Families_In_The_Wild/Database/FIDs/', f_mids='mid.csv'):
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

    print("{} families are being processed".format(len(fid_list)))
    # Load MID LUT for all FIDs.
    df_mids = db.load_mids(dirs_fid, f_csv=f_mids)

    # Load relationship matrices for all FIDs.
    # df_relationships = load_relationship_matrices(dirs_fid, f_csv=f_rel_matrix)
    brothers = []
    for i, fid in enumerate(fid_list):
        # ids = [i for i, s in enumerate(genders) if 'Male' in s]
        # rel_mat = np.array(df_relationships[i])
        rel_mat = db.parse_relationship_matrices(df_mids[i])
        genders = list(df_mids[i].Gender)
        # zero out female subjects
        rel_mat = db.specify_gender(rel_mat, genders, 'm')

        brother_ids = np.where(rel_mat == 2)
        if len(brother_ids[0]) == 0:
            continue
        if not helpers.check_npairs(len(brother_ids[1]), kind, fid):
            print('Error Pair' + fid)
            continue
        # add to list of brothers
        brothers = db.set_pairs(brothers, brother_ids, kind, fid)

    return brothers


def parse_sisters(dir_data='/Users/josephrobinson//Dropbox/Families_In_The_Wild/Database/FIDs/', f_mids='mid.csv'):
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
    dirs_fid, fid_list = load_fids(dir_data)

    print("{} families are being processed".format(len(fid_list)))
    # Load MID LUT for all FIDs.
    df_mids = db.load_mids(dirs_fid, f_csv=f_mids)

    # # Load relationship matrices for all FIDs.
    # df_relationships = load_relationship_matrices(dirs_fid, f_csv=f_rel_matrix)
    sisters = []
    for i, fid in enumerate(fid_list):
        # ids = [i for i, s in enumerate(genders) if 'Male' in s]
        rel_mat = db.parse_relationship_matrices(df_mids[i])
        genders = list(df_mids[i].Gender)

        # zero out female subjects
        rel_mat = db.specify_gender(rel_mat, genders, 'f')

        sister_ids = np.where(rel_mat == 2)

        if not helpers.check_npairs(len(sister_ids[1]), kind, fid):
            continue

        # add to list of brothers
        sisters = db.set_pairs(sisters, sister_ids, kind, fid)

    return sisters


def group_child_parents(ch_ids):
    """
    Group children with parents. This is done by using dictionaries with values as lists (i.e., parents MIDs) and keys
    being MID of their child.
    :param ch_ids:
    :return: List of lists, each list contains a tuple pair
    """
    groups = defaultdict(list)

    for obj in ch_ids:
        groups[obj[0]].append(obj)

    return groups


def tri_subjects(dir_data='/Users/josephrobinson//Dropbox/Families_In_The_Wild/Database/FIDs/', f_mids='mid.csv'):
    """
    Parse sister pairs by referencing member ID LUT and relationship matrix.
def accumulate(l):
    it = itertools.groupby(l, accumulate(operator.itemgetter(0)))
    for key, subiter in it:
        # print(subiter[0])
        yield key, (item[1] for item in subiter)
    Siblings RID is 2 and sister, of course, are Females. Thus, these are the factors we use to identify pairs.

    :param dir_data:        Directory containing folders of FIDs (e.g., F0001/, ..., F????/).
    :param f_mids:
    :return:
    """

    # family directories
    dirs_fid, fid_list = load_fids(dir_data)

    print("{} families are being processed".format(len(fid_list)))
    # Load MID LUT for all FIDs.
    df_mids = db.load_mids(dirs_fid, f_csv=f_mids)

    # # Load relationship matrices for all FIDs.
    # df_relationships = load_relationship_matrices(dirs_fid, f_csv=f_rel_matrix)
    fms = []
    fmd = []
    kind = 'parents-child'
    for i, fid in enumerate(fid_list):
        print(fid)
        # ids = [i for i, s in enumerate(genders) if 'Male' in s]
        # rel_mat = db.parse_relationship_matrices(df_mids[i])
        rel_mat = db.parse_relationship_matrices(df_mids[i])
        genders = list(df_mids[i].Gender)

        # indices of matrix containing 4 or 1; that the matrix is inversed across the diagonal
        # mat_ids = np.where(rel_mat == 1) and np.where(rel_mat.T == 1), np.where(rel_mat == 1) and np.where(
        #     rel_mat.T == 4)

        c_ids = np.where(rel_mat == 1)
        p_ids = np.where(rel_mat == 4)

        if len(list(c_ids)) == 0:
            logger.warn("No pair of parents for child in {}.".format(fid))
            # print("Two parents are not present for child")
            continue

        if len(set(p_ids[1]).__xor__(set(c_ids[0]))) or len(set(c_ids[1]).__xor__(set(p_ids[0]))):
            logger.error("Unmatched pair in {}.".format(fid))
            # print("Unmatched pair")
            continue
        ch_ids = [(p1, p2) for p1, p2 in zip(list(c_ids[0]), list(c_ids[1]))]

        cp_pairs = group_child_parents(ch_ids)

        for cid, pids in cp_pairs.items():
            # pars = rows[np.where(cols == cc)]
            if len(pids) != 2:
                if len(pids) > 2:
                    logger.error("{} parents in {}. {}".format(len(pids), fid, pids))
                    continue
                    # warn.warn("Three parents")
                else:
                    continue
            try:
                p_genders = [genders[pids[0][1]], genders[pids[1][1]]]

            except IndexError:
                print()
            if "m" in p_genders[0] and "f" in p_genders[1]:
                pars_ids = (pids[0][1] + 1, pids[1][1] + 1)
            elif "m" in p_genders[1] and "f" in p_genders[0]:
                pars_ids = pids[1][1] + 1, pids[0][1] + 1
            else:
                logger.error("Parents of same gender in {}. {}".format(fid, pids))
                continue
                # warn.warn("Parents are of same gender for ", fid)

            cmid = "{}/MID{}".format(fid, cid + 1)
            fmid = "{}/MID{}".format(fid, pars_ids[0])
            mmid = "{}/MID{}".format(fid, pars_ids[1])

            if "Male" in genders[cid]:
                fms.append((fmid, mmid, cmid))
            else:
                fmd.append((fmid, mmid, cmid))

    return fmd, fms


def parse_parents(dir_data, f_mids='mid.csv'):
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

    print("{} families are being processed".format(len(fid_list)))
    # Load MID LUT for all FIDs.
    df_mids = db.load_mids(dirs_fid, f_csv=f_mids)

    # Load relationship matrices for all FIDs.
    # df_relationships = load_relationship_matrices(dirs_fid, f_csv=f_rel_matrix)
    fd = []
    fs = []
    md = []
    ms = []
    kind = 'parent-child'
    for i, fid in enumerate(fid_list):
        # ids = [i for i, s in enumerate(genders) if 'Male' in s]
        # rel_mat = np.array(df_relationships[i])
        rel_mat = db.parse_relationship_matrices(df_mids[i])
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
            logger.error("Number of children and parents are different {}.".format(fid))
            # warn.warn()

        if not helpers.check_npairs(len(c_ids[0]) + len(p_ids[0]), kind, fid):
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
            if 'm' in p_gender:
                # fathers
                if 'm' in c_gender:
                    # son
                    md.append(db.Pair(mids=(p_mid, c_mid), fid=fid, kind='fs'))
                else:
                    # daughter
                    ms.append(db.Pair(mids=(p_mid, c_mid), fid=fid, kind='fd'))
            else:
                # mothers
                if 'm' in c_gender:
                    fd.append(db.Pair(mids=(p_mid, c_mid), fid=fid, kind='ms'))
                else:
                    fs.append(db.Pair(mids=(p_mid, c_mid), fid=fid, kind='md'))

    return fd, fs, md, ms


def parse_grandparents(dir_data='/Users/josephrobinson//Dropbox/Families_In_The_Wild/Database/FIDs/',
                       f_mids='mid.csv'):
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

    print("{} families are being processed".format(len(fid_list)))
    # Load MID LUT for all FIDs.
    # Load MID LUT for all FIDs.
    df_mids = db.load_mids(dirs_fid, f_csv=f_mids)

    # Load relationship matrices for all FIDs.
    # df_relationships = load_relationship_matrices(dirs_fid, f_csv=f_rel_matrix)
    gfgd = []
    gfgs = []
    gmgd = []
    gmgs = []
    kind = 'parent-child'
    for i, fid in enumerate(fid_list):
        # ids = [i for i, s in enumerate(genders) if 'Male' in s]
        rel_mat = db.parse_relationship_matrices(df_mids[i])
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

        if not helpers.check_npairs(len(c_ids[0]), kind, fid):
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
            if 'f' in p_gender:
                # fathers
                if 'f' in c_gender:
                    # son
                    gfgs.append(db.Pair(mids=(p_mid, c_mid), fid=fid, kind='gmgd'))
                else:
                    # daughter
                    gfgd.append(db.Pair(mids=(p_mid, c_mid), fid=fid, kind='gmgs'))
            else:
                # mothers
                if 'f' in c_gender:
                    gmgs.append(db.Pair(mids=(p_mid, c_mid), fid=fid, kind='gfgd'))
                else:
                    gmgd.append(db.Pair(mids=(p_mid, c_mid), fid=fid, kind='gfgd'))

    return gfgd, gfgs, gmgd, gmgs


def prepare_fids(dir_fid=dir_home() + "/Dropbox/Families_In_The_Wild/Database/Ann/FW_FIDs/",
                 dirs_out=dir_home() + "/Dropbox/Families_In_The_Wild/Database/FIDs/",
                 do_save=False):
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
    dfs_fams = []
    for fid in fid_dicts:
        # for each fid (i.e., iterate keys of dictionary)
        # print(fid)
        df_rel_mat = fid_dicts[fid]
        col_gender = df_rel_mat.Gender

        # get MIDs
        ids = [i for i, c in enumerate(col_gender) if '-1' not in c]
        tmp = list(range(0, len(ids)))

        same_size, same_contents = helpers.compare_mid_lists(tmp, ids)

        if not (same_size and same_contents):
            logger.error("MIDs and row indices of relationship table differ in {}.".format(fid))
        mids = list(np.array(ids) + 1)
        cols = df_rel_mat.columns[mids]
        # df_rel_mat.loc[ids][cols]

        rel_mat = np.array(df_rel_mat.loc[ids][cols])
        success, messages = helpers.check_rel_matrix(rel_mat, fid=fid)
        if not success:
            logger.error("Relationship matrix failed inspection {}.".format(fid))
            [logger.error("\t{}".format(m)) for m in messages]
            continue

        genders = [g for g in list(col_gender[ids])]
        names = list(df_rel_mat.Name[ids])

        df_fam = pd.DataFrame({'MID': mids})
        df_fam = df_fam.join(df_rel_mat.loc[ids][cols])
        df_fam = df_fam.join(pd.DataFrame({'Gender': genders, 'Name': names}))
        if do_save:
            df_fam.to_csv(dirs_out + "/" + fid + "/mid.csv", index=False)
        dfs_fams.append(df_fam)
    return dfs_fams


if __name__ == '__main__':

    out_bin =  dir_home() + "/master-version/fiwdb/Pairs/"
    dir_fids = dir_home() + "/master-version/fiwdb/FIDs/"
    dir_fid = dir_home() + "/master-version/fiwdb/Ann/FW_FIDs/"

    dir_families = '/master-version/fiwdb/FIDs/'
    fams = fiw.load_families(dir_families)
    logger.info("Output Bin: {}\nFID folder: {}\n Anns folder: {}".format(out_bin, dir_fids, dir_fid))

    do_sibs = True
    do_save = True
    parse_fids = False
    logger.info("Parsing siblings: {}\nSaving Pairs: {}\n Parse FIDs: {}".format(do_sibs, do_save, parse_fids))
    if parse_fids:
        df_fam = prepare_fids(dir_fid=dir_fid, dirs_out=dir_fids)
    if do_sibs:
        print("Parsing Brothers")
        bros = parse_brothers(dir_data=dir_fids)
        print(len(bros))
        for index in range(0, 5):
            print(str(bros[index]))

        pair_set = db.Pairs(bros, kind='brothers')

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

        pair_set = db.Pairs(sis, kind='sisters')

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

        pair_set = db.Pairs(sibs, kind='siblings')

        df_all_faces = get_face_pairs(dir_fids, pair_set.df_pairs)
        if do_save:
            pair_set.write_pairs(out_bin + "sibs-pairs.csv")
            df_all_faces.to_csv(out_bin + 'sibs-faces.csv', index=False)

        del sibs, pair_set, df_all_faces
    # if False:
        print("Parsing Grandparents")
        gfgd, gfgs, gmgd, gmgs = parse_grandparents(dir_data=dir_fids)
        fd_set = db.Pairs(gfgd, kind='gfgd')
        df_all_faces = get_face_pairs(dir_fids, fd_set.df_pairs)
        fd_set.write_pairs(out_bin + "gfgd-pairs.csv")
        df_all_faces.to_csv(out_bin + 'gfgd-faces.csv', index=False)
        print(len(df_all_faces))
        del df_all_faces

        fs_set = db.Pairs(gfgs, kind='gfgs')
        df_all_faces = get_face_pairs(dir_fids, fs_set.df_pairs)
        fs_set.write_pairs(out_bin + "gfgs-pairs.csv")
        df_all_faces.to_csv(out_bin + 'gfgs-faces.csv', index=False)
        print(len(df_all_faces))
        del df_all_faces

        md_set = db.Pairs(gmgd, kind='gmgd')
        df_all_faces = get_face_pairs(dir_fids, md_set.df_pairs)
        md_set.write_pairs(out_bin + "gmgd-pairs.csv")
        df_all_faces.to_csv(out_bin + 'gmgd-faces.csv', index=False)
        print(len(df_all_faces))
        del df_all_faces

        ms_set = db.Pairs(gmgs, kind='gmgs')
        df_all_faces = get_face_pairs(dir_fids, ms_set.df_pairs)
        ms_set.write_pairs(out_bin + "gmgs-pairs.csv")
        df_all_faces.to_csv(out_bin + 'gmgs-faces.csv', index=False)
        print(len(df_all_faces))

        # if False:
        print("Parsing Parents")
        fd, fs, md, ms = parse_parents(dir_data=dir_fids)

        fd_set = db.Pairs(fd, kind='fd')
        df_all_faces = get_face_pairs(dir_fids, fd_set.df_pairs)
        fd_set.write_pairs(out_bin + "fd-pairs.csv")
        df_all_faces.to_csv(out_bin + 'fd-faces.csv', index=False)
        print(len(df_all_faces))
        del df_all_faces

        fs_set = db.Pairs(fs, kind='fs')
        df_all_faces = get_face_pairs(dir_fids, fs_set.df_pairs)
        fs_set.write_pairs(out_bin + "fs-pairs.csv")
        df_all_faces.to_csv(out_bin + 'fs-faces.csv', index=False)
        print(len(df_all_faces))
        del df_all_faces

        md_set = db.Pairs(md, kind='md')
        df_all_faces = get_face_pairs(dir_fids, md_set.df_pairs)
        md_set.write_pairs(out_bin + "md-pairs.csv")
        df_all_faces.to_csv(out_bin + 'md-faces.csv', index=False)
        print(len(df_all_faces))
        del df_all_faces

        ms_set = db.Pairs(ms, kind='ms')
        df_all_faces = get_face_pairs(dir_fids, ms_set.df_pairs)
        ms_set.write_pairs(out_bin + "ms-pairs.csv")
        df_all_faces.to_csv(out_bin + 'ms-faces.csv', index=False)
        print(len(df_all_faces))
        del df_all_faces

            # perpare_fids(dir_fid=dir_fid, dirs_out=dir_fids)
            # print()
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