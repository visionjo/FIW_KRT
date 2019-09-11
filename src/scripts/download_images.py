from __future__ import print_function
import pandas as pd
import numpy as np
import urllib.request
import fiwtools.utils.io as io
import os
f_csv = "/Users/josephrobinson/Dropbox/Families_In_The_Wild/Database/FIW_PIDs.csv"

dir_out = "/Users/josephrobinson/Dropbox/Families_In_The_Wild/Database/ImagesDB/"
df = pd.read_csv(f_csv)

fids = df.FID.unique()

for fid in fids:
    ids = np.where(df.FID == fid)[0]

    dout = dir_out + fid + "/"
    io.mkdir(dout)
    for id in range(len(ids)):
        pid = df.PID[ids[id]]
        url = df.URL[ids[id]]
        fout = dout + pid + ".jpg"
        print(fout)
        path = urllib.request.urlretrieve(url)
        print(path)
        os.rename(path[0], fout)
        # print('2', path)


