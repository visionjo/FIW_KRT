#!/usr/bin/python

# Script to parse different pair types for kin verification.
# TODO refactor

import fiwtools.data.fiw as fiw
import fiwtools.fiwdb.database as db
import fiwtools.utils.log as log

from pyfiw.configs import CONFIGS
logger = log.setup_custom_logger(__name__, f_log='fiw_error_new.log', level=log.INFO)

# def parse(dir_fids, kind, message="", do_save=False, file_prefix=""):
#     """
#       Function to find pairs for specific type, load and return all face pairs.
#     :param dir_fids:
#     :param kind:
#     :param message:
#     :param do_save:
#     :param file_prefix:
#     :return:
# TODO function to replace each block of code below
#     """

# log = logging.getLogger(__name__)
from fiwtools import utils as io

out_bin = CONFIGS.path.dpairs

io.mkdir(out_bin)
dir_fids = CONFIGS.path.dfid
# dir_fid = io.sys_home() + "/Dropbox/Families_In_The_Wild/Database/Ann/FW_FIDs/"
logger.info(f"Output Bin: {out_bin}\nFID folder: {dir_fids}")
do_sibs = True
do_parent_child = True
do_gparent_gchild = True
prepare_fids = False
do_save = True
logger.info(
    f"Parsing siblings: {do_sibs}\nSaving Pairs: {do_save}\n Parse FIDs: {prepare_fids}"
)


if prepare_fids:
    dir_fid = None
    df_fam = fiw.prepare_fids(dir_fid=dir_fid, dirs_out=dir_fids, do_save=do_save)
if do_sibs:
    print("Parsing Brothers")
    bros_pairs = fiw.parse_brothers(dir_data=dir_fids, logger=logger)
    print(len(bros_pairs))
    for index in range(5):
        print(bros_pairs[index])

    pair_set = db.Pairs(bros_pairs, kind='brothers')

    df_all_faces = fiw.get_face_pairs(dir_fids, pair_set.df_pairs)

    if do_save:
        pair_set.write_pairs(f"{out_bin}bb-pairs.csv")
        df_all_faces.to_csv(f'{out_bin}bb-faces.csv', index=False)

    del bros_pairs, pair_set, df_all_faces

    print("Parsing Sisters")
    sis_pairs = fiw.parse_sisters(dir_data=dir_fids, logger=logger)
    print(len(sis_pairs))
    for index in range(5):
        print(sis_pairs[index])

    pair_set = db.Pairs(sis_pairs, kind='sisters')

    df_all_faces = fiw.get_face_pairs(dir_fids, pair_set.df_pairs)

    if do_save:
        pair_set.write_pairs(f"{out_bin}ss-pairs.csv")
        df_all_faces.to_csv(f'{out_bin}ss-faces.csv', index=False)

    del sis_pairs, pair_set, df_all_faces

    print("Parsing Siblings")
    sibs = fiw.parse_siblings(dir_data=dir_fids, logger=logger)
    print(len(sibs))
    for index in range(5):
        print(sibs[index])

    pair_set = db.Pairs(sibs, kind='siblings')

    df_all_faces = fiw.get_face_pairs(dir_fids, pair_set.df_pairs)
    if do_save:
        pair_set.write_pairs(f"{out_bin}sibs-pairs.csv")
        df_all_faces.to_csv(f'{out_bin}sibs-faces.csv', index=False)

    del sibs, pair_set, df_all_faces
if do_gparent_gchild:
    print("Parsing Grandparents")
    gfgd, gfgs, gmgd, gmgs = fiw.parse_grandparents(dir_data=dir_fids, logger=logger)
    fd_set = db.Pairs(gfgd, kind='gfgd')
    df_all_faces = fiw.get_face_pairs(dir_fids, fd_set.df_pairs)

    if do_save:
        fd_set.write_pairs(f"{out_bin}gfgd-pairs.csv")
        df_all_faces.to_csv(f'{out_bin}gfgd-faces.csv', index=False)
    print(len(df_all_faces))
    del df_all_faces

    fs_set = db.Pairs(gfgs, kind='gfgs')
    df_all_faces = fiw.get_face_pairs(dir_fids, fs_set.df_pairs)
    if do_save:
        fs_set.write_pairs(f"{out_bin}gfgs-pairs.csv")
        df_all_faces.to_csv(f'{out_bin}gfgs-faces.csv', index=False)
    print(len(df_all_faces))
    del df_all_faces

    md_set = db.Pairs(gmgd, kind='gmgd')
    df_all_faces = fiw.get_face_pairs(dir_fids, md_set.df_pairs)
    if do_save:
        md_set.write_pairs(f"{out_bin}gmgd-pairs.csv")
        df_all_faces.to_csv(f'{out_bin}gmgd-faces.csv', index=False)
    print(len(df_all_faces), "Face Pairs")
    del df_all_faces

    ms_set = db.Pairs(gmgs, kind='gmgs')
    df_all_faces = fiw.get_face_pairs(dir_fids, ms_set.df_pairs)
    ms_set.write_pairs(f"{out_bin}gmgs-pairs.csv")
    df_all_faces.to_csv(f'{out_bin}gmgs-faces.csv', index=False)
    print(len(df_all_faces), "Face Pairs")

if do_parent_child:
    print("Parsing Parents")
    fd, fs, md, ms = fiw.parse_parents(dir_data=dir_fids, logger=logger)

    fd_set = db.Pairs(fd, kind='fd')
    df_all_faces = fiw.get_face_pairs(dir_fids, fd_set.df_pairs)
    if do_save:
        fd_set.write_pairs(f"{out_bin}fd-pairs.csv")
        df_all_faces.to_csv(f'{out_bin}fd-faces.csv', index=False)
    print(len(df_all_faces), "Face Pairs")
    del df_all_faces

    fs_set = db.Pairs(fs, kind='fs')
    df_all_faces = fiw.get_face_pairs(dir_fids, fs_set.df_pairs)
    if do_save:
        fs_set.write_pairs(f"{out_bin}fs-pairs.csv")
        df_all_faces.to_csv(f'{out_bin}fs-faces.csv', index=False)
    print(len(df_all_faces))
    del df_all_faces

    md_set = db.Pairs(md, kind='md')
    df_all_faces = fiw.get_face_pairs(dir_fids, md_set.df_pairs)
    if do_save:
        md_set.write_pairs(f"{out_bin}md-pairs.csv")
        df_all_faces.to_csv(f'{out_bin}md-faces.csv', index=False)
    print(len(df_all_faces), "Face Pairs")
    del df_all_faces

    ms_set = db.Pairs(ms, kind='ms')
    df_all_faces = fiw.get_face_pairs(dir_fids, ms_set.df_pairs)

    if do_save:
        ms_set.write_pairs(f"{out_bin}ms-pairs.csv")
        df_all_faces.to_csv(f'{out_bin}ms-faces.csv', index=False)
    print(len(df_all_faces), "Face Pairs")
    del df_all_faces