#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
# date: 06/28/17
# name: Joseph Robinson
# description: A set of tools (wraper functions) for pycaffe.
"""
from __future__ import print_function

import scipy.misc
import caffe
import os
from warnings import warn
import numpy as np
import pickle
import src.common.utilities as utils
import src.common.image as imutils
import csv
import scipy.io


########################################################################################################################
###                                                                                                                  ###
###                                              Blob (data) flow                                                    ###
###                                                                                                                  ###
########################################################################################################################
def inspect_blobs(net):
    """
    Function that prints blob information/ data flow wrt net passed in (i.e., LAYER NAME : LAYER TYPE (No. BLOBS)).

    :param net: Network to print blob information of
    :return: None
    """
    print("Inspecting Blobs:")
    for name, blob in net.blobs.iteritems():
        print("{:<5}:  {}".format(name, blob.data.shape))


########################################################################################################################
###                                                                                                                  ###
###                                                    General                                                       ###
###                                                                                                                  ###
########################################################################################################################
def read_lines(in_file):
    """
    Simply reads entire contents of file
    :param in_file:  file to read
    :return:    list object containing lines of file
    """
    with open(in_file) as f:
        return f.readlines()


def binary2mat(bin_file, mat_file):
    """
    Simple function that converts a bin file into a MAT file.

    :param bin_file:    binary file to convert to mat file (input)
    :param mat_file:    mat file to write to (output)
    :return:            None
    """
    proto_data = open(bin_file, "rb").read()
    a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
    mean = caffe.io.blobproto_to_array(a)[0]
    scipy.io.savemat(mat_file, dict(mean=mean), do_compression=True, oned_as='row')


########################################################################################################################
###                                                                                                                  ###
###                                                    Functions                                                     ###
###                                                                                                                  ###
########################################################################################################################
def softmax(x):
    """
    Compute softmax values for each sets of scores in x.

    :param x: input vector
    :return: softmax output for vector x
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


########################################################################################################################
###                                                                                                                  ###
###                                                    Network                                                       ###
###                                                                                                                  ###
########################################################################################################################
def make_prediction_vgg(net, f_image, output_layer='prob'):
    """
    Classify image and return output from specified level [default 'prob')

    :param net: network to use for classification
    :param f_image: file path of image to classify

    :param output_layer: name of layer to output from
    :return:      output of layer
    """
    img = load_prepare_image_vgg(f_image)
    if not img:
        return None

    # read and prepare image
    net.forward_all(data=img)
    output = net.blobs[output_layer].data[0]

    return output


def make_prediction_res(net, f_image, output_layer='fc1'):
    """
    Classify image and return output from specified level [default 'prob')
    :param net: network to use for classification
    :param f_image:   filepath to image
    :param output_layer: name of layer to output from
    :return:      output of layer
    """

    img = load_prepare_image_res(f_image)
    if not img:
        return None

    net.set_input_arrays(np.array(img, order='C').astype(np.float32), np.ones(1, dtype='float32'))
    out = net.forward()
    # get feats of last fully-connected
    return softmax(out[output_layer][0])


# end predict()

########################################################################################################################
###                                                                                                                  ###
###                                                    Images                                                        ###
###                                                                                                                  ###
########################################################################################################################
def load_image(f_image):
    """
    Reads image and prepares for passing to CNN (i.e., caffe format).

    :param f_image:  path to image to be processed
    :return:        loaded image
    :rtype: [numpy.ndarray]
    """
    if not os.path.isfile(f_image):
        warn('caffe_tool.load_prepare_image(): image file was not loaded.', f_image, '\nExit() ')
        return None

    return caffe.io.load_image(f_image)


# end load_image()

def is_caffe_format(image):
    """
    Determines whether image is in caffe format or not. This is done by checking whether image is tensor.

    :param image:       image to check
    :return:            true/false pertaining to whether or not image is tensor (caffe formatted)
    """

    if image.shape[0] == 1 or len(image.shape) == 4:
        # if formatted for single image batch mode or 4 channels
        return True
    else:
        return False


def caffe2rgb(image):
    """
    Converts image to RGB if in caffe format. See is_caffe_format()
    :param image:       image to check

    :return:            rgb image
    """

    if is_caffe_format(image):
        # if tensor, then remove first channel and transpose image from BGR-> RGB
        return image[0, :, :, :].transpose((1, 2, 0))
    else:
        # assuming image is already of type RGB
        return image


def load_prepare_image_res(f_image, avg=np.array([122.782, 117.001, 104.298])):
    """
    Load and prepare image for STResNet Model.

    :param f_image:   fpath to image to be processed
    :param avg:     mean image (default set to that of STResNet)
    :return:        Loaded and preprocessed image
    """
    # rescale from [0, 1] to [0, 255] & convert RGB->BGR
    img = load_image(f_image)
    if img is None:
        print("WARNING")
        return None

    img = img * 255.0  # scale to range 0-255
    img = img - avg  # subtract mean (numpy takes care of dimensions :)
    img = img.transpose((2, 0, 1))  # convert channels to BRG (i.e., per caffe)
    img = img[None, :]  # add singleton dimension shape(3, 224, 224)-> shape(1, 3, 224, 224)

    return img


def load_prepare_resnet_centerloss(f_image, im_dims=[112, 96], avg=127.5):
    """
    Load and prepare image for VGG16 (i.e., VGGFace) Model.

    :param f_image:    fpath to image to be processed
    :param avg:     mean image (default set to that of VGGFace)
    :return:        Loaded and preprocessed image
    """
    # rescale from [0, 1] to [0, 255] & convert RGB->BGR

    img = load_image(f_image)

    if img is None:
        return None

    img = img[:, :, ::-1] * 255.0  #
    img = img - avg  # subtract mean (numpy takes care of dimensions :)
    img /= 128.0
    img = scipy.misc.imresize(img, (112, 96, 3))

    # img = imutils.reshape(img, (96, 112))
    img = img.transpose((2, 0, 1))
    img = img[None, :]  # add singleton dimension

    return img

def load_prepare_image_vgg(f_image, avg=np.array([129.1863, 104.7624, 93.5940])):
    """
    Load and prepare image for VGG16 (i.e., VGGFace) Model.

    :param f_image:    fpath to image to be processed
    :param avg:     mean image (default set to that of VGGFace)
    :return:        Loaded and preprocessed image
    """
    # rescale from [0, 1] to [0, 255] & convert RGB->BGR
    img = load_image(f_image)
    if img is None:
        return None

    img = img[:, :, ::-1] * 255.0  #
    img = img - avg  # subtract mean (numpy takes care of dimensions :)
    img = img.transpose((2, 0, 1))
    img = img[None, :]  # add singleton dimension

    return img


def load_image_mean_shift(f_in, avg=np.array([129.1863, 104.7624, 93.5940]), net_type='vgg'):
    """load and subtract mean from image"""

    img = load_image(f_in)

    if img is None:
        warn('BBNet.load_prepare_image(): image file was not loaded. Exit() ')
        return None
    if net_type.__eq__('vgg'):
        img = img[:, :, ::-1] * 255.0
    else:
        img = img * 255.0

    return img - avg  # subtract mean (numpy takes care of dimensions :)


# def caffe2rgb(image):
#     """Check whether image is formatted for caffe (i.e., tensor); if so, converts to RGB"""
#     if is_caffe_format(image)

########################################################################################################################
###                                                                                                                  ###
###                                         Class Labels & Scores                                                    ###
###                                                                                                                  ###
########################################################################################################################
def read_scores(csv_file):
    """

    :param csv_file: csv file containing scores to load
    :return: loaded scores
    :rtype: [numpy.array]
    """

    """Reads CSV file containing scores of classifier."""
    scores = []
    with open(csv_file) as f:
        contents = csv.reader(f, delimiter=',')
        for row in contents:
            val = float(row[0].strip())
            scores.append(val)
    return np.array(scores)


def top_k_class_indices(score_vector, k=1, get_scores=False):
    """
    Return name corresponding to class label of top prediction score.

    :param score_vector:   vector to determine top scores of
    :param k:              Rank to return (i.e., number of indices to return) [default 1]
    :param get_scores:     return scores along with the indices
    :return:               indices of top k scores; (optionally) return corresponding scores.
    """

    ids = sorted(range(len(score_vector)), key=lambda x: score_vector[x])
    ids = ids[::-1]  # reverse order to be in descending order
    top_k_ids = np.asarray(ids[0:k])

    if get_scores:
        score_vector[::-1].sort()
        top_scores = score_vector[0:k]
        return top_k_ids, top_scores
    else:
        return top_k_ids


def get_class_labels(f_labels):
    """
    Return name list (i.e., class labels).

    :param f_labels:  file of labels to read in
    :return:          class labels (i.e., contents of f_labels)
    """

    with open(f_labels) as fid:
        labels = fid.readlines()

    def clean_labels(x):
        return x.rstrip()

    return list(map(clean_labels, labels))


def read_pickle_labels(pfile):
    """Read labels for STResNet in Pickle file format."""

    with pickle.load(open(pfile)) as f:
        items = f.items()

    values = [i[0] for i in items]
    keys = [int(i[1]) for i in items]

    d = {key: value for (key, value) in zip(keys, values)}
    return d


########################################################################################################################
###                                                                                                                  ###
###                                                 Mean Image                                                       ###
###                                                                                                                  ###
########################################################################################################################
def read_bin(f_binary):
    """
    Function to read binary mean image provided by ResNet

    :param f_binary file path to binary mean image
    :return         binary mean image formatted as RGB
    """

    if not os.path.isfile(f_binary):
        warn('CaffeWrapper.read_bin(): binary file does not exit: ' + f_binary)
        return []

    proto_data = open(f_binary, "rb").read()

    a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)

    avg = caffe.io.blobproto_to_array(a)[0]
    return np.moveaxis(avg, 0, -1)


def check_avg_type(s_avg):
    """
    Function to determine if avg provided in configs is numerical or file path.

    :param s_avg: config ('avg') for CaffeWrapper (1 and 2)
    :return avg array (or image)
    """
    if os.path.isfile(s_avg):
        f_obj = open(s_avg, 'rb')
        avg = pickle.load(f_obj)
    else:
        sp_avg = s_avg.replace('[', '').replace(']', '').split(',')
        if utils.is_number(sp_avg[0]):
            avg = np.array([float(sp_avg[0]), float(sp_avg[1]), float(sp_avg[2])])
        else:
            avg = s_avg
    return avg


########################################################################################################################
###                                                                                                                  ###
###                                                    Layers                                                        ###
###                                                                                                                  ###
########################################################################################################################
def print_layer_info(net):
    """
    Simple function that prints each layer in order (i.e., LAYER NAME : LAYER TYPE (No. BLOBS))
    :param net: Network to print info of
    :return: None
    """
    print("Network layers:")
    for name, layer in zip(net._layer_names, net.layers):
        print("{:<7}: {:17s}({} blobs)".format(name, layer.type, len(layer.blobs)))


def get_network_name(f_in):
    """
    Returns name of network (as listed in prototxt file)

    :param f_in: prototext file path

    :return: name of network
    :rtype: [string]
    """
    lines = read_lines(f_in)
    first_row = lines[0].split()
    net_type = first_row[0].replace("'", "").replace('"', '')
    value = first_row[1].replace("'", "").replace('"', '')

    if net_type == "name":
        return value
    else:
        warn('Name of network not found. Exit("")')
        return ""


def find_layer(lines):
    """
    Find the layer type.
    :param lines:
    :return: reference to layer type
    """
    layer_name = []
    for l in lines:
        if 'type' in l:
            _, layer_name = l.split()
            break
    return layer_name


def find_layer_name(lines):
    """
    Find the layer name

    :param lines:
    :return:
    """
    layer_name = None
    top_name = None
    flag_count = 0
    first_line = lines[0]
    assert first_line.split()[1] is '{', 'Something is wrong'
    brack_count = 1
    for l in lines[1:]:
        if '{' in l:
            brack_count += 1
        if '}' in l:
            brack_count -= 1
        if brack_count == 0:
            break
        if 'name' in l and brack_count == 1:
            flag_count += 1
            _, layer_name = l.split()
            layer_name = layer_name[1:-1]
        if 'top' in l and brack_count == 1:
            flag_count += 1
            _, top_name = l.split()
            top_name = top_name[1:-1]

    assert layer_name is not None, 'no name of a layer found'
    return layer_name, top_name


def get_layers(net):
    """
    Get the layer names of the network.

    :param net: caffe network
    :type net: caffe.Net
    :return: layer names
    :rtype: [string]
    """

    return [layer for layer in net.params.keys()]


########################################################################################################################
###                                                                                                                  ###
###                                                 Visualization                                                    ###
###                                                                                                                  ###
########################################################################################################################
# def visualize_kernels(net, layer, zoom=5):
#     """
#     Visualize kernels in the given layer.
#
#     :param net: caffe network
#     :type net: caffe.Net
#     :param layer: layer name
#     :type layer: string
#     :param zoom: the number of pixels (in width and height) per kernel weight
#     :type zoom: int
#     :return: image visualizing the kernels in a grid
#     :rtype: numpy.ndarray
#     """
#
#     assert layer in get_layers(net), "layer %s not found" % layer
#
#     num_kernels = net.params[layer][0].data.shape[0]
#     num_channels = net.params[layer][0].data.shape[1]
#     kernel_height = net.params[layer][0].data.shape[2]
#     kernel_width = net.params[layer][0].data.shape[3]
#
#     image = numpy.zeros((num_kernels * zoom * kernel_height, num_channels * zoom * kernel_width))
#     for k in range(num_kernels):
#         for c in range(num_channels):
#             kernel = net.params[layer][0].data[k, c, :, :]
#             kernel = cv2.resize(kernel, (zoom * kernel_height, zoom * kernel_width), kernel, 0, 0, cv2.INTER_NEAREST)
#             kernel = (kernel - numpy.min(kernel)) / (numpy.max(kernel) - numpy.min(kernel))
#             image[k * zoom * kernel_height:(k + 1) * zoom * kernel_height,
#             c * zoom * kernel_width:(c + 1) * zoom * kernel_width] = kernel
#
#     return image

#
# def visualize_weights(net, layer, zoom=2):
#     """
#     Visualize weights in a fully conencted layer.
#
#     :param net: caffe network
#     :type net: caffe.Net
#     :param layer: layer name
#     :type layer: string
#     :param zoom: the number of pixels (in width and height) per weight
#     :type zoom: int
#     :return: image visualizing the kernels in a grid
#     :rtype: numpy.ndarray
#     """
#
#     assert layer in get_layers(net), "layer %s not found" % layer
#
#     weights = net.params[layer][0].data
#     weights = (weights - numpy.min(weights)) / (numpy.max(weights) - numpy.min(weights))
#     return cv2.resize(weights, (weights.shape[0] * zoom, weights.shape[1] * zoom), weights, 0, 0, cv2.INTER_NEAREST)


########################################################################################################################
###                                                                                                                  ###
###                                           WhiteBox (to merge)                                                    ###
###                                                                                                                  ###
########################################################################################################################
def Init_Caffe(use_gpu=False, deviceID=0):
    if use_gpu is True:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    caffe.set_device(deviceID)

    return


def Load_Network(dataset):
    if dataset == 'VGG-Face':
        ModelDir = '/proj/loki/bdecann/loki/sandbox/bdecann/whitebox/' + \
                   'VGG-Face/'
        net_model_path = os.path.join(ModelDir,
                                      'VGG_FACE_WITHOUT_SOFTMAX_deploy.prototxt')
        net_weights_path = os.path.join(ModelDir, 'VGG_FACE.caffemodel')
    else:
        raise ValueError('Test dataset not recognized')

    return caffe.Net(net_model_path, net_weights_path, caffe.TEST)


def GetBlobNames(net):
    return list(net._blob_names)


def Predict(net, I, flag=0):
    # Pass in Image and return either the weights (e.g., scores)
    # at the last blob or the label

    L = GetBlobNames(net)
    net.blobs[L[0]].reshape(1, I.shape[0], I.shape[1], I.shape[2])
    net.blobs[L[0]].data[...] = I

    # Forward Pass
    out = net.forward()[L[-1]].flatten()

    if flag == 0:
        # Return the label
        return np.argmax(out)
    else:
        # Return the weights / scores
        return out


def batch_predict(net, Imgs):
    # Pass in a set of images and return the predicted labels
    # for each image

    L = GetBlobNames(net)

    net.blobs[L[0]].reshape(Imgs.shape[0], Imgs.shape[1],
                            Imgs.shape[2], Imgs.shape[3])
    net.blobs[L[0]].data[...] = Imgs

    out = net.forward()[L[-1]]

    return out.argmax(1)


def backward_pass(net, dzdy, source_layer_index=-1, target_layer_index=0):
    # Pass in data for backpropagation. Default is the last layer (source)
    # to the first layer (target)
    #
    # Note: Currently only supports last layer to target layer
    B = GetBlobNames(net)
    L = list(net._layer_names)

    net.blobs[B[source_layer_index]].diff[...] = dzdy

    if target_layer_index == 0:
        out = net.backward()
    else:
        out = net.backward(end=L[target_layer_index])

    return out[B[target_layer_index]]
