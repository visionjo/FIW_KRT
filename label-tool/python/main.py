import glob
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import tqdm
from sklearn.svm import SVC

def load_features(din, key='n*'):
    """
    Load features for directory and its subdirectory identifiable via key
    :param din: directory with features and subdirectory
    :param key: to identify using wildcard (i.e., name or pattern of subdirectory)
    :return:
    """
    files_feats_outer = glob.glob(din + '*.pkl')

    if len(files_feats_outer) == 0:
        # skip if no faces to compare with

        return [], []
    # get inner features
    files_feats_inner = glob.glob(din + key + '/*.pkl')

    return {f.split('/')[-1].replace('.pkl', ''): pd.read_pickle(f) for f in files_feats_outer}, \
           {f.split('/')[-1].replace('.pkl', ''): pd.read_pickle(f) for f in files_feats_inner}

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

threshold_sim_score = 0.32
# params
dir_features = '/home/jrobby/master-version/fiwdb/fiw_plus_vgg-cropped-features2/'
dir_faces = '/home/jrobby/master-version/fiwdb/fiw_plus_vgg-cropped2/'
dir_out = '/home/jrobby/master-version/fiwdb/fiw_plus_vgg-clustered/'
if not os.path.isdir(dir_out):
    os.mkdir(dir_out)

# get names of inner directories
fids = [f.split('/')[-1] for f in glob.glob(dir_faces + 'F????')]
fids.sort()

mids = ["/".join(f.split('/')[-2:]) for f in glob.glob(dir_faces + 'F????/MID*')]
mids.sort()

dirs_in = [dir_features + f + '/' for f in mids]
print('Processing {} subjects'.format(len(dirs_in)))
##
#  PART 1
##
for i, dir_in in enumerate(tqdm.tqdm(dirs_in)):
    # load features
    feats_orig, feats_new = load_features(dir_in)

    if len(feats_orig) == 0:
        print('no original faces for {}'.format(mids[i]))
        continue
    # apply metric to features
    scores = cosine_similarity(list(feats_orig.values()), list(feats_new.values()))

    # compare inner-most directory with root, determine if same individual
    mean_scores = scores.mean(axis=1)

    if np.median(mean_scores) > 0.35:
        # if same, clean face data free of non-faces and faces of other persons
        feats = feats_orig
        # Output a boolean label and confidence score for each face
        for key, val in feats_new.items():
            feats[key] = val

        # save all features
        pd.to_pickle(feats, dir_out + mids[i].replace('/', '_') + '.pkl')

        # determine which faces being added are true matches
        mean_scores = scores.mean(axis=0)

        mask = (threshold_sim_score < mean_scores).astype(np.uint8)

        df = pd.DataFrame(np.array([list(feats_new.keys()), mask, mean_scores]).T,
                          columns=['file', 'label', 'score'])

        df.to_csv( dir_out + mids[i].replace('/', '_') + '-labels' + '.csv', index=False)

##
#  PART 2
##
feature_files = glob.glob(dir_out + '*.pkl')

print("{} subjects assumed to be the same".format(len(feature_files)))

# save results in format compatible with GUI

# for each subject, filter out non-matched faces

# provide labels: 1 if true match; else 0

# write CSV to disc

# prepare to pass to JAVA tool