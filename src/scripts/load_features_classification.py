import pickle

import fiwtools.data.fiw as fiw
import pandas as pd

file_fold = ['0.csv', '1.csv', '2.csv', '3.csv', '4.csv']
file_out = ['0.pkl', '1.pkl', '2.pkl', '3.pkl', '4.pkl']
dir_info = '/Volumes/JROB/Families_In_The_Wild/database/classification/info/'
dir_feats = '/Volumes/JROB/Families_In_The_Wild/database/feats/fc7/'

# split_info = io.readtxt(dir_info + file_fold)

for i, fin in enumerate(file_fold):

    if i == 0:
        continue
    split_info_pd = pd.read_csv(dir_info + fin, delimiter=',')

    files_feats = [dir_feats + f for f in split_info_pd['Var1']]

    feats = fiw.load_all_features(dir_feats, split_info_pd['Var1'])
    with open(dir_info + file_out[i], "wb") as pickling_on:
        pickle.dump(feats, pickling_on)

#
# pd.read_table()
#
# fh = open(dir_info + file_fold, 'r')
# print(fh.readlines())