#!/usr/bin/python
#   Script to detect faces in images of FIW. Detections are named <PID>face<No. Detected>.jpg
#
#   Faces are detected using dlib's CNN face detector.
#
#   This example shows how to run a CNN based face detector using dlib.  The
#   example loads a pretrained model and uses it to find faces in images.  The
#   CNN model is much more accurate than the HOG based model shown in the
#   face_detector.py example, but takes much more computational power to
#   run, and is meant to be executed on a GPU to attain reasonable speed.
#
#   You can download the pre-trained model from:
#       http://dlib.net/files/mmod_human_face_detector.dat.bz2
#
#   The examples/faces folder contains some jpg images of people.  You can run
#   this program on them and see the detections by executing the
# TODO merge with other dlib face detection script-- add flag to save jpegs and csv
from __future__ import print_function

import fiwtools.utils.io as myio
import dlib
from skimage import io
import glob
import pandas as pd


print("DLIB's CNN FACE DETECTOR: START")
dir_root = "Families_In_The_Wild/Database/"
dir_images = dir_root + "Images/"
dir_det_out = dir_root + "F0664/"

# instantiate CNN face detector
f_model = "~/Documents/dlib-19.6/mmod_human_face_detector.dat"
cnn_face_detector = dlib.cnn_face_detection_model_v1(f_model)

dir_fids = glob.glob(dir_images + "F0664/")
dir_fids.sort()

f_pids = glob.glob(dir_images + "F0664/P*.jpg")
f_pids.sort()

f_prefix = [f.replace(dir_images, "").replace("/", "_").replace(".jpg", "_face") for f in f_pids]

# f_pids =list( np.array(f_pids)[ids])
fids = [myio.file_base(myio.filepath(p)) for p in f_pids]
pids = [myio.file_base(p) for p in f_pids]
# f_prefix =list( np.array(f_prefix)[ids])
npids = len(f_pids)
print("Processing {} images".format(npids))

# images = [io.imread(f) for f in f_pids]

# Second argument indicates that we should upsample the image 1 time (i.e., bigger image to detect of more faces).
dets = [cnn_face_detector(io.imread(f), 1) for f in f_pids]

df = pd.DataFrame(columns=['FID', 'PID', 'face_id', 'filename', 'left', 'top', 'right', 'bottom', 'confidence'])

# dets = cnn_face_detector(images, 1)

# '''
# Detector returns a mmod_rectangles object containing a list of mmod_rectangle objects, which are accessed by
#  iterating over the mmod_rectangles object. mmod_rectangle has 2 members, dlib.rectangle object & confidence score.
#
# It is possible to pass a list of images to the detector.
#     - like this: dets = cnn_face_detector([image list], upsample_num, batch_size = 128)
# In this case it will return a mmod_rectangless object.
# This object behaves just like a list of lists and can be iterated over.
# '''
predictor_path = "shape_predictor_68_face_landmarks.dat"
sp = dlib.shape_predictor(predictor_path)
print("Number of faces detected: {}".format(len(dets)))
counter = 0
for faces, prefix in zip(dets, f_prefix):
    img = io.imread(counter)
    for i, d in enumerate(faces):
        f_name = prefix + str(i)
        df.loc[counter] = [fids[counter], pids[counter], i, f_name, d.rect.left(), d.rect.top(), d.rect.right(),
                           d.rect.bottom(), d.confidence]
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
            i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))
        shape = sp(img, d)
        dlib.save_face_chip(img, shape, dir_det_out + f_name + ".jpg")

    counter += 1

df.to_csv("dnn_face_detections_bb_2.csv")
