import common.io as io
import glob
import pandas as pd
import database.fiw as fiw
import common.log as log
import numpy as np


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
        for x, y, z in zip(faces1, faces2, faces3):
            all_faces.append([x, y, z])
            # for y in faces2:
            #     for z in faces3:
            #         all_faces.append([x, y, z])
                # [[x, y] for x, y in zip(faces1, faces2)]
    arr_all_face = np.array(all_faces)
    return pd.DataFrame({header[0]: arr_all_face[:, 0], header[1]: arr_all_face[:, 1], header[2]: arr_all_face[:, 2]})
    # arr_pairs = np.array(all_faces)
    #
    # print('No. Face Pairs is {}.'.format(arr_pairs.shape[0]))
    # return pd.DataFrame({"p1": arr_pairs[:, 0], "p2": arr_pairs[:, 1], "p3": arr_pairs[:, 2]})


logger = log.setup_custom_logger(__name__, f_log='tri-info.log', level=log.INFO)

out_bin = io.sys_home() + "/Dropbox/Families_In_The_Wild/Database/Pairs/"
dir_fids = io.sys_home() + "/Dropbox/Families_In_The_Wild/Database/FIDs/"


do_save = False
logger.info("Parsing Tri-Subject Pairs:\n\t{}\n\t{}\n".format(out_bin, dir_fids))

fmd, fms = fiw.tri_subjects(dir_data=dir_fids)
logger.info("{}".format(out_bin, dir_fids))
# pair_set.write_pairs(out_bin + "sibs-pairs.csv")
# df_all_faces.to_csv(out_bin + 'sibs-faces.csv', index=False)
if do_save:
    fiw.write_list_tri_pairs(out_bin + "fmd-pairs.csv", fmd)
    fiw.write_list_tri_pairs(out_bin + "fms-pairs.csv", fms)
print(len(fmd))
for index in range(0, 5):
    print(str(fmd[index]))
print(len(fms))
for index in range(0, 5):
    print(str(fms[index]))

df_all_faces = combine_face_pairs(dir_fids, fmd, header=('F', 'M', 'D'))
df_all_faces.to_csv(out_bin + 'fmd-faces.csv', index=False)
logger.info("Parsing Tri-Subject Pairs:\n\t{}\n\t{}\n".format(out_bin, dir_fids))

del df_all_faces

df_all_faces = combine_face_pairs(dir_fids, fms, header=('F', 'M', 'S'))
df_all_faces.to_csv(out_bin + 'fms-faces.csv', index=False)



