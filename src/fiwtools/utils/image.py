from __future__ import print_function
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

import fiwtools.utils.io as io
import fiwtools.utils.common as utils
import csv
import matplotlib.image as mpimg
import tempfile
import urllib
from skimage.transform import resize

# import urllib.request

__author__ = 'Joseph Robinson'


def bgr2rgb(im_bgr):
    """Wrapper for opencv color conversion color BGR (opencv format) to color RGB"""
    return cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)


def crop_face(img, cropping):
    if cropping is not None:
        img = img[cropping[1]:cropping[3], cropping[0]:cropping[2], :]
        print('> Cropping with: ', cropping)
    else:
        print('> No Cropping')
    return img


def crop_fun(frontal_raw, crop_model):
    frontal_raw = crop_face(frontal_raw, crop_model)
    return frontal_raw


def gray2bgr(im_gray):
    """Wrapper for opencv color conversion grayscale to color BGR (opencv format)"""
    return cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR)


def gray2jet(img):
    """[0,1] grayscale to [0.255] RGB"""
    jet = plt.get_cmap('jet')
    return np.uint8(255.0 * jet(img)[:, :, 0:3])


def list(imdir):
    """return list of images with absolute path in a directory"""
    return [os.path.abspath(os.path.join(imdir, item)) for item in os.listdir(imdir) if
            (io.is_img(item) and not io.is_hidden_file(item))]


def saveimage(f_image, image):
    """ Save image"""
    # check that folder of image file exists; else, make it
    io.mkdir(io.parent_dir(f_image))

    cv2.imwrite(f_image, image)


def savelist(imdir, outfile):
    """Write out all images in a directory to a provided file with each line containing absolute path to image"""
    return io.writelist(list(imdir), outfile)


def read(img_file):
    """Read image from file
    :param img_file: filepath of image to open and return
    :type img_file: string
    """
    if not io.is_file(img_file):
        return None
    return mpimg.imread(img_file)


def reshape(img, im_dims, rgb=True):
    """
    Resize dimensions of image; else, preserve rgb channels by default, or set False to convert to gray then resize.

    :param img:         image to resize :param im_dims:     dimension (h, w) to set img to :param rgb:
    boolean flag to specify whether rgb should be returned. Note that if gray is passed, then function will convert
    to RGB by copying single channel across 3 channels. :return:
    """
    import pdb
    pdb.set_trace()
    if rgb:
        return resize(img.reshape, im_dims[0], im_dims[1], 3)
    else:
        return resize(rgb2bgr(img).reshape([im_dims[0], im_dims[1]]))


def resize(img):
    """

    :param img:
    :return:
    """
    img = np.true_divide(img - np.amin(img), np.amax(img) - np.amin(img))

    return img


def rgb2bgr(im_rgb):
    """Wrapper for opencv color conversion color RGB to color BGR (opencv format)"""
    return cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)


def rgb2gray(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    return 0.299 * R + 0.587 * G + 0.114 * B


def rgb_to_csv(img, ofile):
    """Write 3-channel numpy matrix (i.e., RGB img) to csv, with each column
    containing a single channel and each channel vectorized with elements
    ordered down columns"""

    width, height, channel = img.shape
    with open(ofile, 'w+') as f:
        f.write('R,G,B\n')
        # read the details of each pixel and write them to the file
        for x in range(width):
            for y in range(height):
                r = img[x, y][0]
                g = img[x, y][1]
                b = img[x, y][2]
                f.write('{0},{1},{2}\n'.format(r, g, b))


def temp(ext='jpg'):
    """Create a temporary image with the given extension"""
    if ext[0] == '.':
        ext = ext[1:]
    return tempfile.mktemp() + '.' + ext


def temp_png():
    """Create a temporay PNG file"""
    return temp('png')


def url_to_image(url):
    """
  a helper function that downloads the image, converts it to a NumPy array,
  and then reads it into OpenCV format
  """
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # return the image
    return image


def writejet(img, imfile=None):
    """Write a grayscale numpy image as a jet colormapped image to the given file"""
    if imfile is None:
        imfile = temp_png()

    if utils.is_numpy(img):
        if img.ndim == 2:
            cv2.imwrite(imfile, rgb2bgr(gray2jet(img)))
        else:
            raise ValueError('Input must be a 2D numpy array')
    else:
        raise ValueError('Input must be numpy array')
    return imfile


def writegray(img, imfile=None):
    """Write a floating point grayscale numpy image as [0,255] grayscale"""
    if imfile is None:
        imfile = temp_png()
    if utils.is_numpy(img):
        if img.dtype == np.dtype('uint8'):
            # Assume that uint8 is in the range [0,255]
            cv2.imwrite(imfile, img)
        elif img.dtype == np.dtype('float32'):
            # Convert [0,1.0] to uint8 [0,255]
            cv2.imwrite(imfile, np.uint8(img * 255.0))
        else:
            raise ValueError('Unsupported datatype - Numpy array must be uint8 or float32')
    else:
        raise ValueError('Input must be numpy array')
    return imfile


def write(img, imfile=None, writeas=None):
    """Write a floating point 2D numpy image as jet or gray, 3D numpy as rgb or bgr"""
    if imfile is None:
        imfile = temp_png()
    if not utils.is_numpy(img):
        raise ValueError('image must by numpy object')
    if writeas is None:
        if img.ndim == 2:
            writeas = 'gray'
        else:
            writeas = 'bgr'

    if writeas in ['jet']:
        writejet(img, imfile)
    elif writeas in ['gray']:
        writegray(img, imfile)
    elif writeas in ['rgb']:
        if img.ndim != 3:
            raise ValueError('numpy array must be 3D')
        if img.dtype == np.dtype('uint8'):
            cv2.imwrite(imfile, rgb2bgr(img))  # convert to BGR
        elif img.dtype == np.dtype('float32'):
            cv2.imwrite(imfile, rgb2bgr(np.uint8(255.0 * img)))  # convert to uint8 then BGR
    elif writeas in ['bgr']:
        if img.ndim != 3:
            raise ValueError('numpy array must be 3D')
        if img.dtype == np.dtype('uint8'):
            cv2.imwrite(imfile, img)  # convert to BGR
        elif img.dtype == np.dtype('float32'):
            cv2.imwrite(imfile, (np.uint8(255.0 * img)))  # convert to uint8 then BGR
    else:
        raise ValueError('unsupported writeas')

    return imfile


def normalize(img):
    """Normalize image to have no negative numbers"""
    imin = np.min(img)
    imax = np.max(img)

    return (img - imin) / (imax - imin)


def get_image_list(in_file):
    f_images = []
    with open(in_file) as f:
        contents = csv.reader(f, delimiter=',')
        for x, row in enumerate(contents):
            if x == 0:
                continue
            val = row[0].strip()
            f_images.append(val)
    return f_images
