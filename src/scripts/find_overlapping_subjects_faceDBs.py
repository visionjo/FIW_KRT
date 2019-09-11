from collections import defaultdict

import pandas as pd
import numpy as np
import re
from fiwtools.utils import io
from fiwtools.fiwdb import database as db

def read_mscelebnames(f_names):
    names = io.readtxt(f_names)
    ns = [n.split()[1:] for n in names]
    nss = []
    for n in ns:
        if len(n) > 1:
            nname = ""
            for nn in n:
                nname += " " + nn.split("@")[0]
            nss.append(nname)
        else:
            print(str(n).split("@")[0])
            nss.append(str(n).split("@")[0])
    return nss


def clean_text(s):
    """ Remove non alphabetic characters. E.g. 'B:a,n+a1n$a' becomes 'Banana' """

    s = re.sub("[^a-z A-Z _]", "", s)
    s = s.replace(' n ', ' ')
    return s.strip().lower()

f_namelist = '/Volumes/Seagate Backup Plus Drive/msceleb1m20180924/name.en'
f_list = '/Users/josephrobinson/Desktop/people.txt'
f_fids = '/Users/josephrobinson/Dropbox/Families_In_The_Wild/Database/FIW_FIDs.csv'

dir_families = '/Users/josephrobinson/Dropbox/Families_In_The_Wild/Database/FIDs_NEW/'

fid_dirs = db.load_fids(dir_families)
dict_names = db.get_names_dictionaries(dir_families, fid_dirs[1])
fids_df = pd.read_csv(f_fids, delimiter='\t')
names = pd.read_csv(f_list)

name_list = [clean_text(r).lower() for r in names['10']]
# nlist = [r.split('_')[0] for r in name_list]
nlist = [r.lower().split('_') for r in name_list if len(r) > 0]
nitems = len(fids_df)

fids_tup = [(fids_df.iloc[r]['FIDs'], fids_df.iloc[r]['surnames'].lower().split('.')) for r in range(nitems)]
fids_dict = dict(fids_tup)
fids = list(fids_dict.keys())

surnames={}
for k, v in fids_dict.items():
    surnames[k] = v[0]

allnames = []
allfids = []
counter = 0
for i, fid in enumerate(fids):
    ns = dict_names[fid]
    allfids.append((fid, surnames[fid]))
    for n in ns:
        allnames.append((surnames[fid], n.lower()))
        counter += 1


celebnames = read_mscelebnames(f_namelist)
celebnames = [clean_text(n).strip().lower() for n in celebnames]

clist = [(n.split()[-1], n.split()[:-1]) for n in celebnames if len(n) > 1]

cnlist = []
for cc in clist:
    for c in cc[1]:
        cnlist.append((cc[0], c))

# flist = [r.lower().split('.') for r in fids_df['surnames']]

ids = np.zeros(len(allnames))

ddict = defaultdict(list)
for j, fid in enumerate(allnames):

    # ddict[fid[0]].append()
    for i, n in enumerate(cnlist):
        if fid[0] == n[0]:
            # for k, nn in enumerate(name):
            if n[1] == fid[1]:
                # print(nn, ' : ', fids['surnames'][j], " : ", name_list[i])
                print(n, ' : ', fid, " : ", cnlist[i])



ids = np.zeros(len(allnames))

for j, fid in enumerate(dict_names.keys()):
    names = [n.lower() for n in dict_names[fid]]
    surname = list(fids_df['surnames'][fids_df['FIDs'] == fid])
    if len(surname) == 0:
        continue
    surname = surname[0]
    name = surname.lower().split('.')
    for i, n in enumerate(nlist):
        if name[0] == n[-1].lower():
            for k, nn in enumerate(name):
                if n[0] == nn:
                    # print(nn, ' : ', fids['surnames'][j], " : ", name_list[i])
                    print(nn, ' : ', surname, " : ", nlist[i])


