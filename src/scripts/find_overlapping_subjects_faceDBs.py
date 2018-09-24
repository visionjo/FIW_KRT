import pandas as pd
import numpy as np
import re

from src.fiwdb import database as db


def clean_text(s):
    """ Remove non alphabetic characters. E.g. 'B:a,n+a1n$a' becomes 'Banana' """

    s = re.sub("[^a-z A-Z _]", "", s)
    s = s.replace(' n ', ' ')
    return s


f_list = '/Users/josephrobinson/Desktop/people.txt'
f_fids = '/Users/josephrobinson/Dropbox/Families_In_The_Wild/Database/FIW_FIDs.csv'

dir_families = '/Users/josephrobinson/Dropbox/Families_In_The_Wild/Database/FIDs_NEW/'

fid_dirs = db.load_fids(dir_families)
dict_names = db.get_names_dictionaries(dir_families, fid_dirs[1])
fids = pd.read_csv(f_fids, delimiter='\t')
names = pd.read_csv(f_list)

name_list = [clean_text(r).lower() for r in names['10']]
# nlist = [r.split('_')[0] for r in name_list]
nlist = [r.lower().split('_') for r in name_list if len(r) > 0]

flist = [r.lower().split('.') for r in fids['surnames']]
ids = np.zeros(len(flist))

for j, fid in enumerate(dict_names.keys()):
    names = [n.lower() for n in dict_names[fid]]
    surname = list(fids['surnames'][fids['FIDs'] == fid])
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
