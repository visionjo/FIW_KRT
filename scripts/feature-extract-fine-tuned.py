# Script to extract features from different layers of VGG faces
# docker run -ti -v /home/jrobby/Dropbox/Families_In_The_Wild/Database/FIDs:/data -v /home/jrobby/Dropbox/Families_In_The_Wild/python/models:/models -v /media/jrobby/Seagate\ Backup\ Plus\ Drive1/FIW_dataset/FIW_Extended/feats:/feats -v /home/jrobby/Dropbox/github/FIW_KRT:/code  bvlc/caffe:gpu bash

# TODO: PCA
# TODO refactor code, such to enable interface for options, opposed to hard coded as done currently.
import numpy as np
import glob
import common.io as io

import fiwdb.database as fiwdb
import os
import common.log as log
import frameworks.pycaffe.net_wrapper as cw
import frameworks.pycaffe.tools as caffe_tools


logger = log.setup_custom_logger(__name__, f_log='kinwildW-feat-extractor.log', level=log.INFO)


# if __name__ == "__main__":
model_def = '/model/face_deploy.prototxt'
weights = '/model/face_train_test_iter_1600.caffemodel'
input = '/data/KinFaceW-I/images/'
output = '/data/KinFaceW-I/features/fine-tuned/'
io.mkdir(output)
mode = 'cpu'
dims = 200
layer = 'fc5'
types = ['father-dau', 'father-son', 'mother-dau', 'mother-son']
gpu_id = 0
overwrite = False

layers = ['fc5', 'conv5_5', 'conv5_4']
for l in layers:

    dout = output + "/" + l + "/"
    io.mkdir(dout)
    logger.info("Output Directory: {}\nInput Image Directory: {}\n".format(output, input))

    io.mkdir(dout)
    if mode == 'gpu':
        my_net = cw.CaffeWrapper(model_def=model_def, gpu_id=gpu_id, mode=mode, model_weights=weights, do_init=True)
    else:
        my_net = cw.CaffeWrapper(model_def=model_def, mode=mode, model_weights=weights, do_init=False)
    my_net.init()

    for ptype in types:
        # each pair type
        print("Processing image pairs of type {}".format(ptype))
        # dir_images = []
        dir_images = input + ptype + "/"
        io.mkdir(dout + ptype)
        # dirs_fid, fids = fiwdb.load_fids(dir_images)
        im_files = glob.glob(dir_images + "/*.jpg")
        feat_files = [dout + str(f).replace(input, "").replace(".jpg", ".csv") for f in im_files]
        print("{} faces to extract features from.".format(len(feat_files)))
        # import pdb
        # pdb.set_trace()
        nimages = len(im_files)
        for i in range(nimages):
            in_file = im_files[i]
            f_file = feat_files[i]
            if os.path.exists(f_file):
                continue
            print(f_file)
            # layers = layers
            # ofile = dout + str(ifile).replace(input, "").replace(".jpg", ".csv")
            # if not io.mkdir(io.parent_dir(f_file)):
                # continue
                # continue
            # if os.path.isfile(in_file):

            logger.info("Extracting feature for: {}\n".format(in_file))
            fname = io.file_base(in_file)

            image = caffe_tools.load_prepare_resnet_centerloss(in_file, (112, 96))
            # import pdb
            # pdb.set_trace()
            # my_net.net.blobs['data'].data[...] = image
            output = my_net.net.forward()

            # io.mkdir(io.filepath(in_file))
            feat = my_net.net.blobs[l].data[0]
            logger.info("Writing feature to disk: {}\n".format(f_file))

            np.savetxt(f_file, feat.flatten(), delimiter=',')  # X is an array


#
# # net = net.net
# # # dir_data = '/data/FIDs2/'
# # dir_data = '/data/FIW_Extended/FIDs/'
# # dir_out = '/data/FIW_Extended/feats/vgg_face/'
#
# # fam_dirs = glob.glob(dir_data + "F????/")
# # mems_dirs = [glob.glob(d + "MID*" + '/') for d in fam_dirs]
#
# # obins = [d1[6::] for d1 in d, for d in mems_dirs]
#
# # impaths = [glob.glob(d1 + "*.jpg") for d1 in d for d in mems_dirs]
# #
#
# # obins = [dir_out + l + "/" for l in layers]
# for x, f_dir in enumerate(fam_dirs):
#     print(x, "/", len(f_dir))
#     m_dirs = mems_dirs[x]
#     for y, mdir in enumerate(m_dirs):
#         ipaths = glob.glob(mdir + "*.jpg")
#
#         # mid = utils.IO.file_base(mdir)
#         # os.mkdir(mdir)
#         # tmp = dir_out + layers[0] + "/" + f_dir[12::] + mdir[18::]
#         # if os.path.isdir(tmp):
#         #     continue
#         print("Member", y)
#         for ipath in ipaths:
#
#             fname = utils.IO.file_base(ipath)
#
#
#
#
#             ### perform classification
#
#
#             for j, l in enumerate(layers):
#                 # d_out = dir_out + l + "/" + f_dir[12::] + mdir[18::]
#                 d_out = obins[j] + "/" + mdir[24::]
#                 utils.IO.mkdir(d_out)
#                 # os.makedirs(d_out)
#                 f_out = d_out + fname + '.csv'
#


    # image = net.load_image(f_image)
    # feat = net.extract_features(image, output_layer='avg_pool2')
    # feat = feat[0]
    # # import pdb
    # # pdb.set_trace()pwd
    # # with open(f_out, "w") as output_file:
    # np.savetxt(f_out, feat, delimiter=',')  # X is an array
    # pickle.dump(feat, output_file)


# pickle.dump(feat, f_out)

                    # feature





# sub_dirs = ['father-dau', 'father-son', 'mother-dau', 'mother-son']
#
# dir_root = '/media/jrobby/Seagate Backup Plus Drive1/DATA/Kinship/KinFaceW-II/'
# dir_features = dir_root + 'vgg_face/'
#
# dir_results = dir_features + 'results_spca/'
# io.mkdir(dir_results)
#
# dir_perms = dir_root + 'perm/'
# dir_lists = dir_root + 'meta_data/'
#
# do_pca = True
# k = 200
# # load experimental settings for 5-fold verification
# f_lists = glob.glob(dir_lists + "*.csv")
# pair_types = [io.file_base(f) for f in f_lists]
#
# dir_feats = [dir_features + p + "/" for p in sub_dirs]
#
# fold_list = [1, 2, 3, 4, 5]
# for ids in [1]:
#     folds, labels, pairs1, pairs2 = kinwild.read_pair_list(f_lists[ids])
#
#     d_out = dir_results + pair_types[ids] + "/"
#     io.mkdir(d_out)
#
#     for feature in features:
#         print("processing features from layer", feature)
#         feats1 = kinwild.load_all_features(dir_feats[ids] + feature + "/", pairs1)
#         feats2 = kinwild.load_all_features(dir_feats[ids] + feature + "/", pairs2)
#         dir_result = d_out + feature + "/"
#         io.mkdir(dir_result)
#         ts_matches = []
#         sim = []
#         accs = np.zeros((5,))
#         for jj, fold in enumerate(fold_list):
#             print("Fold", fold, "/", len(fold_list))
#             train_ids = fold != folds
#             test_ids = fold == folds
#
#             train_labels = labels[train_ids]
#             test_labels = labels[test_ids]
#             # pairs 1
#             train_feats1 = feats1[train_ids]
#             test_feats1 = feats1[test_ids]
#
#             # pairs 2
#             train_feats2 = feats2[train_ids]
#             test_feats2 = feats2[test_ids]
#             feats = []
#             for f1, f2 in zip(train_feats1[train_labels == 1], train_feats1[train_labels == 1]):
#                 feats.append(f1)
#                 feats.append(f2)
#
#             # all_feats = {**tuple(trfeats1), **tuple(trfeats2)}
#             print("Normalizing Features")
#             feat_vecs = np.array(feats)
#             std_scale = skpreprocess.StandardScaler().fit(feat_vecs)
#
#             # normalize test set
#             X_feats_1 = std_scale.transform(test_feats1)
#             X_feats_2 = std_scale.transform(test_feats2)
#
#             if do_pca:
#                 X_train = std_scale.transform(feat_vecs)
#
#                 print("PCA")
#
#                 try:
#                     # pca [default k is 200]
#                     clf = TruncatedSVD(k)
#                     evecs = clf.fit_transform(X_train.T)
#                 except ValueError:
#                     # if reduced dimensionality is less than or equal to the number of samples
#                     knew = X_train.shape[0] - 1
#                     clf = TruncatedSVD(knew)
#                     evecs = clf.fit_transform(X_train.T)
#
#                 # X_scaled, evals, evecs = futils.pca(X_train)
#
#                 # apply PCA
#                 X_feats_1 = np.dot(evecs.T, X_feats_1.T).T
#                 X_feats_2 = np.dot(evecs.T, X_feats_2.T).T
#
#             cosine_sim = pw.cosine_similarity(X_feats_1, X_feats_2)
#             scores = np.diag(cosine_sim)
#
#             # val_labels = [int(pair[1]) for pair in pair_val_list]
#
#             ts_matches.append(test_labels)
#             sim.append(scores)
#             # Compute ROC curve and ROC area
#             fpr, tpr, _ = roc_curve(test_labels, scores)
#             roc_auc = auc(fpr, tpr)
#             print("Acc:", roc_auc)
#             print('Saving Results')
#             dir_out = dir_result + str(fold) + "/"
#             io.mkdir(dir_out)
#             np.savetxt(dir_out + "fpr.csv", fpr)
#             np.savetxt(dir_out + "tpr.csv", tpr)
#             np.savetxt(dir_out + "roc_auc.csv", [roc_auc])
#             accs[jj] = roc_auc
#         fpr, tpr, _ = roc_curve(np.array(ts_matches).flatten(), np.array(sim).flatten())
#         roc_auc = auc(fpr, tpr)
#         np.savetxt(dir_result + "fpr.csv", fpr) /
#         np.savetxt(dir_result + "tpr.csv", tpr)
#         np.savetxt(dir_result + "roc_auc.csv", [accs.mean()])
#         np.savetxt(dir_result + "acc.csv", [accs.mean()])
