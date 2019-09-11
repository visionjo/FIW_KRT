import numpy as np
import pickle
import tensorflow as tf
import math
from random import gauss
import fileinput
import base64
from numpy import linalg as LA


def pre_label_weight(pre, gnd, pthresh=0.95, coverage='yes', threshold='no'):
    plog = np.amax(pre, axis=1)
    pind = np.argmax(pre, axis=1)
    yind = np.argmax(gnd, axis=1)
    resultList = []
    TotalLabelCnt = 0
    for i in range(pre.shape[0]):
        if pind[i] == yind[i]:
            label = '1'
        else:
            label = '-1'
        weight = 1.0
        resultList.append((plog[i], label, weight))
        TotalLabelCnt = TotalLabelCnt + weight
        resultList.sort(key=lambda x: x[0], reverse=True)

    tp = fp = 0
    coverageList = []
    precisionList = []
    thresholdList = []

    for result in resultList:
        predict = result[0]
        label = result[1]
        weight = result[2]
        if label == '1':
            tp += weight
        if label == '-1':
            fp += weight

        precision = float(tp) / float(tp + fp)
        coverage = float(tp + fp) / TotalLabelCnt

        coverageList.append(coverage)
        precisionList.append(precision)
        thresholdList.append(predict)

    coverage = 0
    threshold_select = 0
    for i in range(len(coverageList) - 1, 0 - 1, -1):
        if precisionList[i] > pthresh:
            coverage = coverageList[i]
            threshold_select = thresholdList[i]
            break

    if (len(coverageList) > 100):
        stepsize = int(np.floor(len(coverageList) / 100))
    else:
        stepsize = 1

    coverageV = coverageList[0:len(coverageList):stepsize]
    coverageV.append(coverageList[len(coverageList) - 1])
    accuracyV = precisionList[0:len(precisionList):stepsize]
    accuracyV.append(precisionList[len(precisionList) - 1])

    return coverageV, accuracyV, coverage, threshold_select


def readfile(path):
    flist = []
    for line in fileinput.input(path):
        line = line.strip('\n')
        flist.append(line)
    return flist


def nextbatch0(pos, shuffle, BATCH_SIZE, fea_tsv, lbl_tsv, feaIdx, lblIdx, num_classes=21000):
    batchIdx = shuffle[pos:pos + BATCH_SIZE]
    fea = []
    lbl = []
    for i in range(BATCH_SIZE):
        linenum = feaIdx[int(batchIdx[i])]
        fea_tsv.seek(int(linenum), 0)
        fline = fea_tsv.readline()
        cols = fline.rstrip().split('\t')
        vecstring = cols[10]
        featureVectorBytes = base64.b64decode(vecstring)
        featureVectorArray = np.frombuffer(featureVectorBytes, dtype=np.float32)
        fea.append(featureVectorArray)
        linenum = lblIdx[int(batchIdx[i])]
        lbl_tsv.seek(int(linenum), 0)
        bline = lbl_tsv.readline()
        cols = bline.rstrip().split('\t')
        lbl.append(cols[1])

    fea = np.array(fea, dtype=np.float32)
    lbl = np.array(lbl, dtype=int)
    lbl = dense_to_one_hot(lbl, num_classes)
    return fea, lbl


def nextbatch(pos, shuffle, BATCH_SIZE, fea_tsv, lbl_tsv, feaIdx, lblIdx, num_classes=21000):
    batchIdx = shuffle[pos:pos + BATCH_SIZE]
    fea = []
    lbl = []
    for i in range(BATCH_SIZE):
        linenum = feaIdx[int(batchIdx[i])]
        fea_tsv.seek(int(linenum), 0)
        fline = fea_tsv.readline()
        cols = fline.rstrip().split('\t')
        vecstring = cols[10]
        featureVectorBytes = base64.b64decode(vecstring)
        featureVectorArray = np.frombuffer(featureVectorBytes, dtype=np.float32)
        fea.append(featureVectorArray)
        linenum = lblIdx[int(batchIdx[i])]
        lbl_tsv.seek(int(linenum), 0)
        bline = lbl_tsv.readline()
        cols = bline.rstrip().split('\t')
        lbl.append(cols[1])

    fea = np.array(fea, dtype=np.float32)
    lbl = np.array(lbl, dtype=int)
    lbl1 = dense_to_one_hot(lbl, num_classes)
    slbl = lbl1[lbl > 19999, :]
    sfea = fea[lbl > 19999, :]
    return fea, lbl1, slbl, sfea


def normalization(data, alp=65):
    for i in range(data.shape[0]):
        data[i, :] = data[i, :] / LA.norm(data[i, :]) * alp
    return data


def white_noise(X):
    X_noise = X.copy()
    n_samples = X.shape[0]
    n_features = X.shape[1]
    for i in range(n_samples):
        series = [gauss(0.0, 0.11) for i in range(n_features)]
        X_noise[i, :] = X[i, :] + series
    return X_noise


## load label map
def LabelMapping(lbl_map_path):
    file = open(lbl_map_path)
    labelmap = {}
    while 1:
        line = file.readline()
        cols = line.rstrip().split('\t')
        if not line:
            break
        labelmap[cols[0]] = cols[1]
    file.close()
    return labelmap


def find_all_index(arr, item):
    return [i for i, a in enumerate(arr) if a == item]


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def data_generate(label_path, data_path, num_classes, repmat):
    pkl_label = open(label_path, 'rb')
    pkl_data = open(data_path, 'rb')
    labels = pickle.load(pkl_label)
    labels_np = np.array(labels, dtype=int)

    data = pickle.load(pkl_data)
    data_np = np.array(data, dtype=np.float32)

    labels_bn = dense_to_one_hot(labels_np, num_classes)
    # labels_bn1 = labels_bn
    # data_np1 = data_np

    if repmat == 1:
        for i in range(1, 8):
            # data_np = white_noise(data_np)#masking_noise(data_np1, 1)
            labels_bn = np.concatenate((labels_bn, labels_bn), axis=0)
            data_np = np.concatenate((data_np, data_np), axis=0)

    pkl_data.close()
    pkl_label.close()
    return data_np, labels_bn


def data_generate2(label_path, data_path, num_classes, repmat):
    pkl_label = open(label_path, 'rb')
    pkl_data = open(data_path, 'rb')
    labels = pickle.load(pkl_label)
    labels_np = np.array(labels, dtype=int)

    data = pickle.load(pkl_data)
    data_np = np.array(data, dtype=np.float32)

    labels_bn = dense_to_one_hot(labels_np, num_classes)
    if repmat == 1:
        for i in range(1, 8):
            labels_bn = np.concatenate((labels_bn, labels_bn), axis=0)
            data_np = np.concatenate((data_np, data_np), axis=0)

    pkl_data.close()
    pkl_label.close()
    return data_np, labels_bn, labels


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


# Leaky Relu
def Leaky_Relu(x, alpha=0., max_value=None):
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=np.float32), tf.cast(max_value, dtype=np.float32))
        x -= tf.constant(alpha, dtype=np.float32) * negative_part
    return x


def Euclidean(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return math.sqrt(((npvec1 - npvec2) ** 2).sum())


def Cosine(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return npvec1.dot(npvec2) / (math.sqrt((npvec1 ** 2).sum()) * math.sqrt((npvec2 ** 2).sum()))
