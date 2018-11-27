#!/usr/bin/python
#   Script to detect faces in images of FIW. Detections are named <PID>face<No. Detected>.jpg
#
#   Faces are detected using dlib's CNN face detector.
#
#
#   You can download the pre-trained model from:
#       http://dlib.net/files/mmod_human_face_detector.dat.bz2
#

import dlib
from skimage import io
import glob
import numpy as np
import pandas as pd
import fiwtools.utils.io as myio

print("DLIB CNN FACE DETECTOR: START")

# ids = range(32 + 1)
dir_root = "/home/jrobby/Dropbox/Families_In_The_Wild/Database/FIW_Extended/"
dir_images = dir_root + "Images/"
dir_det_out = dir_root + "dnn_face_detections/"

# instantiate CNN face detector
f_model = "/home/jrobby/Documents/dlib-19.6/mmod_human_face_detector.dat"
cnn_face_detector = dlib.cnn_face_detection_model_v1(f_model)

dir_fids = glob.glob(dir_images + "F????/")
dir_fids.sort()

f_pids = glob.glob(dir_images + "F????/P*.jpg")

f_pids.sort()

f_prefix = [f.replace(dir_images, "").replace("/", "_").replace(".jpg", "_") for f in f_pids]

f_pids =list( np.array(f_pids)[ids])
fids = [myio.file_base(myio.filepath(p)) for p in f_pids]
pids = [myio.file_base(p) for p in f_pids]
f_prefix =list( np.array(f_prefix)[ids])
npids = len(f_pids)
print("Processing {} images".format(npids))

# images = [io.imread(f) for f in f_pids]

# '''
# Detector returns a mmod_rectangles object containing a list of mmod_rectangle objects, which are accessed by
#  iterating over the mmod_rectangles object. mmod_rectangle has 2 members, dlib.rectangle object & confidence score.
#
# It is possible to pass a list of images to the detector.
#     - like this: dets = cnn_face_detector([image list], upsample_num, batch_size = 128)
# In this case it will return a mmod_rectangless object.
# This object behaves just like a list of lists and can be iterated over.
# '''
# Second argument indicates that we should upsample the image 1 time (i.e., bigger image to detect of more faces).
dets = [cnn_face_detector(io.imread(f), 1) for f in f_pids]

df = pd.DataFrame(columns=['FID', 'PID', 'face_id', 'filename', 'left', 'top', 'right', 'bottom', 'confidence'])




print("Number of faces detected: {}".format(len(dets)))
counter = 0
for faces, prefix in zip(dets, f_prefix):
    # build dataframe of face detections and corresponding metadata
    for i, d in enumerate(faces):
        f_name = prefix + str(i)
        df.loc[counter] = [fids[counter], pids[counter], i,  f_name, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence]
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
            i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))
        rects = dlib.rectangles()
        rects.extend([d.rect for d in dets])

    counter += 1

print(counter, "faces detected")
# write dataframe to CSV
df.to_csv("dnn_face_detections.csv")

print("DLIB's CNN FACE DETECTOR: DONE")
