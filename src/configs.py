from easydict import EasyDict
from src.fiwtools.utils.io import sys_home, is_file, mkdir
import json

CONFIGS = None
f_config = 'logs/configs.json'
if is_file(f_config) and False:
    with open('logs/configs.json', 'r') as fp:
        content = json.load(fp)
    config = EasyDict(content)
else:
    config = EasyDict()

    config.settings = EasyDict()
    config.settings.abbr_gender = ['f', 'm']
    config.settings.abbr_gender.sort()
    config.settings.n_folds = 5
    config.settings.rand_seed = 123
    config.settings.types = ['bb', 'ss', 'sibs', 'fd', 'fs', 'md', 'ms', 'gfgd', 'gfgs', 'gmgd', 'gmgs']

    config.path = EasyDict()
    config.path.dsource = sys_home() + 'master-version/'

    config.path.dbroot = config.path.dsource + 'fiwdb/'
    config.path.dfid = config.path.dbroot + 'FIDs/'


    config.path.dn_nonmid= 'unrelated_and_nonfaces/'
    config.path.data_table = config.path.dbroot + 'datatable.pkl'
    config.path.dfeatures = config.path.dbroot + 'FIDs-features/'
    mkdir(config.path.dfeatures)
    config.path.dimages = config.path.dbroot + 'Images/'
    mkdir(config.path.dimages)

    config.path.features = config.path.dfeatures + 'features.pkl'

    config.path.dlists = config.path.dbroot + 'lists/'
    config.path.dpairs = config.path.dlists + 'pairs/'
    config.path.dlogs = config.path.dbroot + 'logs/'
    config.path.f_log = config.path.dlogs + 'fiw.log'
    mkdir(config.path.dlists)
    mkdir(config.path.dpairs)
    config.path.base_lists = config.path.dlists + 'base/'
    mkdir(config.path.base_lists)

    config.path.master_pairs_list = config.path.dlists + 'merged_pairs.csv'
    config.path.master_pairs_list_pkl = config.path.dlists + 'pair_table.pkl'
    config.path.subject_lut = config.path.dlists + 'subject_lut.csv'
    config.path.subject_lut_pkl = config.path.dlists + 'subject_lut.pkl'
    config.path.image_lut_pkl = config.path.dlists + 'image_lut.pkl'
    config.path.image_lut = config.path.dlists + 'image_lut.csv'

    config.path.fn_mid = 'mid.csv'  # file storing FID labels for each families
    config.path.fn_pid = 'PIDs.csv'  # master PID database file
    config.path.fn_log = 'fiwdb.log'  # output log filename
    config.path.fn_rid = 'FIW_RIDs.csv'  # file name for relationship type look-up (i.e., RID table)
    config.path.fn_fid = 'FIW_FIDs.csv'  # master FID database file
    # config.path.individual_pairs_lists = [config.path.base_lists + t + '_pairs.csv' for t in config.settings.attributes]
    config.path.thresholds_csv = config.path.dbroot + 'thresholds_per_ethnicity.csv'
    config.path.doutput = config.path.dbroot + 'outputs/'
    mkdir(config.path.doutput)
    config.path.outputs = EasyDict()
    config.path.outputs.sdm = config.path.doutput + 'signal_detection_models.pdf'

    config.image = EasyDict()
    config.image.ext = '.jpg'

    mkdir(config.path.dlogs)
    with open(config.path.dlogs + 'configs.json', 'w') as fp:
        json.dump(config, fp)

CONFIGS = EasyDict(config)
