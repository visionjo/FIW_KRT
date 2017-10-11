#!/usr/bin/python

# Script to parse different pair types for kin verification.
# TODO refactor
import database.fiw as fiw
import fiwdb.database as db
import common.log as log

logger = log.setup_custom_logger(__name__)

import logging
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

out_bin = "/Users/josephrobinson/Dropbox/Families_In_The_Wild/Database/Pairs2/"
dir_fids = "/Users/josephrobinson/Dropbox/Families_In_The_Wild/Database/FIDs/"
dir_fid = "/Users/josephrobinson/Dropbox/Families_In_The_Wild/Database/Ann/FW_FIDs/"
logger.info("Output Bin: {}\nFID folder: {}\n Anns folder: {}".format(out_bin, dir_fids, dir_fid))
do_sibs = False
do_parent_child = True
do_gparent_gchild = False
prepare_fids = False
do_save = True
logger.info("Parsing siblings: {}\nSaving Pairs: {}\n Parse FIDs: {}".format(do_sibs, do_save, prepare_fids))

if prepare_fids:
    df_fam = fiw.prepare_fids(dir_fid=dir_fid, dirs_out=dir_fids, do_save=do_save)
if do_sibs:
    print("Parsing Brothers")
    bros_pairs = fiw.parse_brothers(dir_data=dir_fids)
    print(len(bros_pairs))
    for index in range(0, 5):
        print(str(bros_pairs[index]))

    pair_set = db.Pairs(bros_pairs, kind='brothers')

    df_all_faces = fiw.get_face_pairs(dir_fids, pair_set.df_pairs)

    if do_save:
        pair_set.write_pairs(out_bin + "bb-pairs.csv")
        df_all_faces.to_csv(out_bin + 'bb-faces.csv', index=False)

    del bros_pairs, pair_set, df_all_faces

    print("Parsing Sisters")
    sis_pairs = fiw.parse_sisters(dir_data=dir_fids)
    print(len(sis_pairs))
    for index in range(0, 5):
        print(str(sis_pairs[index]))

    pair_set = db.Pairs(sis_pairs, kind='sisters')

    df_all_faces = fiw.get_face_pairs(dir_fids, pair_set.df_pairs)

    if do_save:
        pair_set.write_pairs(out_bin + "ss-pairs.csv")
        df_all_faces.to_csv(out_bin + 'ss-faces.csv', index=False)

    del sis_pairs, pair_set, df_all_faces

    print("Parsing Siblings")
    sibs = fiw.parse_siblings(dir_data=dir_fids)
    print(len(sibs))
    for index in range(0, 5):
        print(str(sibs[index]))

    pair_set = db.Pairs(sibs, kind='siblings')

    df_all_faces = fiw.get_face_pairs(dir_fids, pair_set.df_pairs)
    if do_save:
        pair_set.write_pairs(out_bin + "sibs-pairs.csv")
        df_all_faces.to_csv(out_bin + 'sibs-faces.csv', index=False)

    del sibs, pair_set, df_all_faces
if do_gparent_gchild:
    print("Parsing Grandparents")
    gfgd, gfgs, gmgd, gmgs = fiw.parse_grandparents(dir_data=dir_fids)
    fd_set = db.Pairs(gfgd, kind='gfgd')
    df_all_faces = fiw.get_face_pairs(dir_fids, fd_set.df_pairs)

    if do_save:
        fd_set.write_pairs(out_bin + "gfgd-pairs.csv")
        df_all_faces.to_csv(out_bin + 'gfgd-faces.csv', index=False)
    print(len(df_all_faces))
    del df_all_faces

    fs_set = db.Pairs(gfgs, kind='gfgs')
    df_all_faces = fiw.get_face_pairs(dir_fids, fs_set.df_pairs)
    if do_save:
        fs_set.write_pairs(out_bin + "gfgs-pairs.csv")
        df_all_faces.to_csv(out_bin + 'gfgs-faces.csv', index=False)
    print(len(df_all_faces))
    del df_all_faces

    md_set = db.Pairs(gmgd, kind='gmgd')
    df_all_faces = fiw.get_face_pairs(dir_fids, md_set.df_pairs)
    if do_save:
        md_set.write_pairs(out_bin + "gmgd-pairs.csv")
        df_all_faces.to_csv(out_bin + 'gmgd-faces.csv', index=False)
    print(len(df_all_faces), "Face Pairs")
    del df_all_faces

    ms_set = db.Pairs(gmgs, kind='gmgs')
    df_all_faces = fiw.get_face_pairs(dir_fids, ms_set.df_pairs)
    ms_set.write_pairs(out_bin + "gmgs-pairs.csv")
    df_all_faces.to_csv(out_bin + 'gmgs-faces.csv', index=False)
    print(len(df_all_faces), "Face Pairs")

if do_parent_child:
    print("Parsing Parents")
    fd, fs, md, ms = fiw.parse_parents(dir_data=dir_fids)

    fd_set = db.Pairs(fd, kind='fd')
    df_all_faces = fiw.get_face_pairs(dir_fids, fd_set.df_pairs)
    if do_save:
        fd_set.write_pairs(out_bin + "fd-pairs.csv")
        df_all_faces.to_csv(out_bin + 'fd-faces.csv', index=False)
    print(len(df_all_faces), "Face Pairs")
    del df_all_faces

    fs_set = db.Pairs(fs, kind='fs')
    df_all_faces = fiw.get_face_pairs(dir_fids, fs_set.df_pairs)
    if do_save:
        fs_set.write_pairs(out_bin + "fs-pairs.csv")
        df_all_faces.to_csv(out_bin + 'fs-faces.csv', index=False)
    print(len(df_all_faces))
    del df_all_faces

    md_set = db.Pairs(md, kind='md')
    df_all_faces = fiw.get_face_pairs(dir_fids, md_set.df_pairs)
    if do_save:
        md_set.write_pairs(out_bin + "md-pairs.csv")
        df_all_faces.to_csv(out_bin + 'md-faces.csv', index=False)
    print(len(df_all_faces), "Face Pairs")
    del df_all_faces

    ms_set = db.Pairs(ms, kind='ms')
    df_all_faces = fiw.get_face_pairs(dir_fids, ms_set.df_pairs)

    if do_save:
        ms_set.write_pairs(out_bin + "ms-pairs.csv")
        df_all_faces.to_csv(out_bin + 'ms-faces.csv', index=False)
    print(len(df_all_faces), "Face Pairs")
    del df_all_faces

if True:
    fmd, fms = fiw.tri_subjects(dir_data=dir_fids)
    print(len(fmd))
    for index in range(0, 5):
        print(str(fmd[index]))
    print(len(fms))
    for index in range(0, 5):
        print(str(fms[index]))
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
