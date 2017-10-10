import common.io as io
import database.fiw as fiw
import common.log as log

logger = log.setup_custom_logger(__name__, f_log='tri-info.log', level=log.INFO)

out_bin = io.sys_home() + "/Dropbox/Families_In_The_Wild/Database/Pairs/"
dir_fids = io.sys_home() + "/Dropbox/Families_In_The_Wild/Database/FIDs/"

# logger.info("Output Bin: {}\nFID folder: {}\n Anns folder: {}".format(out_bin, dir_fids, dir_fid))

do_save = False
logger.info("Parsing Tri-Subject Pairs:\n\t{}\n\t{}\n".format(out_bin, dir_fids))

fmd, fms = fiw.tri_subjects(dir_data=dir_fids)
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
