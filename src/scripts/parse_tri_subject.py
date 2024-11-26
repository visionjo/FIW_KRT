from __future__ import print_function
import fiwtools.utils.io as io
import glob
import pandas as pd
import fiwtools.data.fiw as fiw
import fiwtools.utils.log as log
import numpy as np
from fiwtools.utils.io import mkdir

def combine_face_pairs(dirs_fid, tri_pairs, header=('p1', 'p2', 'p3')):
    """ Assumed tuple of 3 MIDs. All combinations of face pairs are generated and returned as DF
    """
    all_faces = []
    for tri_pair in tri_pairs:
        paths1 = glob.glob(dirs_fid + "/" + tri_pair[0] + "/*.jpg")
        paths2 = glob.glob(dirs_fid + "/" + tri_pair[1] + "/*.jpg")
        paths3 = glob.glob(dirs_fid + "/" + tri_pair[2] + "/*.jpg")

        faces1 = [p.replace(dirs_fid + "/", "").replace(dirs_fid, "") for p in paths1]
        faces2 = [p.replace(dirs_fid + "/", "").replace(dirs_fid, "") for p in paths2]
        faces3 = [p.replace(dirs_fid + "/", "").replace(dirs_fid, "") for p in paths3]
        # print(faces1, faces2)
        all_faces.extend([x, y, z] for x, y, z in zip(faces1, faces2, faces3))
    arr_all_face = np.array(all_faces)
    return pd.DataFrame({header[0]: arr_all_face[:, 0], header[1]: arr_all_face[:, 1], header[2]: arr_all_face[:, 2]})
    # arr_pairs = np.array(all_faces)
    #
    # print('No. Face Pairs is {}.'.format(arr_pairs.shape[0]))
    # return pd.DataFrame({"p1": arr_pairs[:, 0], "p2": arr_pairs[:, 1], "p3": arr_pairs[:, 2]})


logger = log.setup_custom_logger(__name__, f_log='tri-info.log', level=log.INFO)

out_bin = io.sys_home() + "/Dropbox/Families_In_The_Wild/Database/tripairs//"
mkdir(out_bin)
dir_fids = io.sys_home() + "/Dropbox/Families_In_The_Wild/Database/FIDs_New/"



do_save = False
logger.info(f"Parsing Tri-Subject Pairs:\n\t{out_bin}\n\t{dir_fids}\n")

fmd, fms = fiw.tri_subjects(dir_data=dir_fids)
logger.info(f"{out_bin}")
# pair_set.write_pairs(out_bin + "sibs-pairs.csv")
# df_all_faces.to_csv(out_bin + 'sibs-faces.csv', index=False)
if do_save:
    fiw.write_list_tri_pairs(out_bin + "fmd-pairs.csv", fmd)
    fiw.write_list_tri_pairs(out_bin + "fms-pairs.csv", fms)
print(len(fmd))
for index in range(5):
    print(fmd[index])
print(len(fms))
for index in range(5):
    print(fms[index])

df_all_faces = combine_face_pairs(dir_fids, fmd, header=('F', 'M', 'D'))
df_all_faces.to_csv(out_bin + 'fmd-faces.csv', index=False)
logger.info(f"Parsing Tri-Subject Pairs:\n\t{out_bin}\n\t{dir_fids}\n")

del df_all_faces

df_all_faces = combine_face_pairs(dir_fids, fms, header=('F', 'M', 'S'))
df_all_faces.to_csv(out_bin + 'fms-faces.csv', index=False)



