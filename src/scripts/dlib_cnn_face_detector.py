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
import pandas as pd
import fiwtools.utils.io as myio
import argparse


if __name__ == '__main__':

    print("DLIB CNN FACE DETECTOR: START")

    parser = argparse.ArgumentParser(description='Face Detector: Save BB coordinate to CSV.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--image_dir', default=myio.sys_home() + "/Dropbox/Families_In_The_Wild/Database/Images/",
                        help='Directory containing image files to run detector on.')
    parser.add_argument('-o', '--output', default=myio.sys_home() + "/Dropbox/Families_In_The_Wild/Database/Database/facesbb_xyhw.csv",
                        help='File to store BB to (CSV)')
    parser.add_argument('-m', '--model_path', default=myio.sys_home() + "/WORK/FIW_KRT/include/mmod_human_face_detector.dat",
                        help="File to dlib mmod trained detector.")
    args = parser.parse_args()

    dir_images = args.image_dir

    # instantiate CNN face detector
    f_model = args.model_path
    cnn_face_detector = dlib.cnn_face_detection_model_v1(f_model)

    dir_fids = glob.glob(dir_images + "F????/")
    dir_fids.sort()

    f_pids = glob.glob(dir_images + "F????/P*.jpg")

    f_pids.sort()

    f_prefix = [f.replace(dir_images, "").replace("/", "_").replace(".jpg", "_") for f in f_pids]

    # f_pids =list(np.array(f_pids)[ids])
    fids = [myio.file_base(myio.filepath(p)) for p in f_pids]
    pids = [myio.file_base(p) for p in f_pids]
    # f_prefix =list( np.array(f_prefix)[ids])
    npids = len(pids)
    print(f"Processing {npids} images")

    # Second argument indicates that we should upsample the image 1 time (i.e., bigger image to detect of more faces).
    dets = [cnn_face_detector(io.imread(f), 1) for f in f_pids]

    df = pd.DataFrame(columns=['FID', 'PID', 'face_id', 'filename', 'left', 'top', 'right', 'bottom', 'confidence'])

    print(f"Number of faces detected: {len(dets)}")
    counter = 0
    for faces, prefix in zip(dets, f_prefix):
        # build dataframe of face detections and corresponding metadata
        for i, d in enumerate(faces):
            f_name = prefix + str(i)
            df.loc[counter] = [fids[counter], pids[counter], i,  f_name, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence]
            print(
                f"Detection {i}: Left: {d.rect.left()} Top: {d.rect.top()} Right: {d.rect.right()} Bottom: {d.rect.bottom()} Confidence: {d.confidence}"
            )

            rects = dlib.rectangles()
            rects.extend([d.rect for d in dets])

        counter += 1

    print(counter, "faces detected")
    # write dataframe to CSV
    df.to_csv(args.output)

    print("DLIB's CNN FACE DETECTOR: DONE")
