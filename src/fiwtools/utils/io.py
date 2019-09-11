from __future__ import print_function

import io
import os
import string
import warnings as warn

import scipy.io as scio
import numpy as np
from .common import is_numpy


def csv_list(imdir):
    """Return a list of absolute paths of *.csv files in current directory"""
    return [os.path.join(imdir, item) for item in os.listdir(imdir) if is_csv(item)]


def dir_list(indir):
    """return list of directories in a directory"""
    return [os.path.abspath(os.path.join(indir, item)) for item in os.listdir(indir) if
            (os.path.isdir(os.path.join(indir, item)) and not is_hidden_file(item))]


def file_base(filename):
    """Return c for filename /a/b/c.ext"""
    (head, tail) = os.path.split(filename)
    (base, ext) = os.path.splitext(tail)
    return base


def file_ext(filename):
    """Given filename /a/b/c.ext return .ext"""
    (head, tail) = os.path.split(filename)
    try:
        parts = string.rsplit(tail, '.', 2)
        if len(parts) == 3:
            ext = '.%s.%s' % (parts[1], parts[2])  # # tar.gz
        else:
            ext = '.' + parts[1]
    except:
        ext = None

    return ext


def parent_dir(filename):
    """Return /a/b for filename /a/b/c.ext"""
    (head, tail) = os.path.split(filename)
    return head


def pklist(imdir):
    """Return a list of absolute paths of *.pk files in current directory"""
    return [os.path.join(imdir, item) for item in os.listdir(imdir) if is_pickle(os.path.join(imdir, item))]


def file_tail(filename):
    """Return c.ext for filename /a/b/c.ext"""
    (head, tail) = os.path.split(filename)
    return tail


def is_img(path):
    """Is object an image with a known extension ['.jpg','.jpeg','.png','.tif','.tiff','.pgm','.ppm','.gif','.bmp']?"""
    (filename, ext) = os.path.splitext(path)
    return ext.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.pgm', '.ppm', '.gif', '.bmp']


def is_pickle(filename):
    """Is the file a pickle archive file"""
    return is_file(filename) and os.path.exists(filename) and file_ext(filename).lower() in ['.pk', '.pkl']


def is_text_file(path):
    """Is the given file a text file?"""
    (filename, ext) = os.path.splitext(path)
    return ext.lower() in ['.txt'] and (filename[0] != '.')


def is_video(path):
    """Is a file a video with a known video extension ['.avi','.mp4','.mov','.wmv','.mpg']?"""
    (filename, ext) = os.path.splitext(path)
    return ext.lower() in ['.avi', '.mp4', '.mov', '.wmv', 'mpg']


def is_csv(path):
    """Is a file a CSV file extension?"""
    (filename, ext) = os.path.splitext(path)
    return ext.lower() in ['.csv', '.CSV']


def is_file(path):
    """Wrapper for os.path.is_file"""
    return os.path.isfile(str(path))


def is_dir(path):
    """Wrapper for os.path.isdir"""
    return os.path.isdir(path)


def is_hidden_file(filename):
    """Does the filename start with a period?"""
    return filename[0] == '.'


def load_mat(matfile):
    return scio.loadmat(matfile)


def readcsv(infile, separator=','):
    """Read a csv file into a list of lists"""
    with open(infile, 'r') as f:
        list_of_rows = [[x.strip() for x in r.split(separator)] for r in f.readlines()]
    return list_of_rows


def readlist(infile):
    """Read each row of file as an element of the list"""
    with open(infile, 'r') as f:
        list_of_rows = [r for r in f.readlines()]
    return list_of_rows


def read_mat(txtfile, delimiter=' '):
    """Whitespace separated values defining columns, lines define rows.  Return numpy array"""
    with open(txtfile, 'rb') as csvfile:
        M = [np.float32(row.split(delimiter)) for row in csvfile]
    return np.array(M)


def readtxt(ifile):
    """ Simple function to read text file and remove clean ends of spaces and \n"""
    with open(ifile, 'r') as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def sys_home():
    """

    :return: Home directory (platform agnostic)
    """
    return os.path.expanduser("~")


def mkdir(output):
    """
    Make directory if does not already exist.
    :param output:
    :return:    True if no directory exists, and 'output' was made; else, False.
    """
    if not os.path.exists(output):
        os.makedirs(output)
        return True
    return False


def filepath(filename):
    """Return /a/b for filename /a/b/c.ext"""
    (head, tail) = os.path.split(filename)
    return head


def newpath(filename, newdir):
    """Return /a/b for filename /a/b/c.ext"""
    (head, tail) = os.path.split(filename)
    return os.path.join(newdir, tail)


def videolist(videodir):
    """return list of images with absolute path in a directory"""
    return [os.path.abspath(os.path.join(videodir, item)) for item in os.listdir(videodir) if
            (is_video(item) and not is_hidden_file(item))]


def writecsv(list_of_tuples, outfile, mode='w', separator=','):
    """Write list of tuples to output csv file with each list element on a row and tuple elements separated by comma"""
    list_of_tuples = list_of_tuples if not is_numpy(list_of_tuples) else list_of_tuples.tolist()
    with open(outfile, mode) as f:
        for u in list_of_tuples:
            n = len(u)
            for (k, v) in enumerate(u):
                if (k + 1) < n:
                    f.write(str(v) + separator)
                else:
                    f.write(str(v) + '\n')
    return outfile


def writelist(mylist, outfile, mode='w'):
    """Write list of strings to an output file with each row an element of the list"""
    with open(outfile, mode) as f:
        for s in mylist:
            f.write(str(s) + '\n')
    return (outfile)


def txtlist(imdir):
    """Return a list of absolute paths of *.txt files in current directory"""
    return [os.path.join(imdir, item) for item in os.listdir(imdir) if io.is_text_file(item) and not io.is_hidden_file(item)]


def check_paths(*paths):
    """
    Function that checks variable number of files (i.e., unordered arguments, *paths). If any of the files do not exist '
    then function fails (i.e., no info about failed indices, but just pass (True) or fail (False))

    :param paths:   unordered args, each pointing to file.
    :return:
    """
    do_exist = True
    for x, path in enumerate(paths):
        if not os.path.isfile(path):
            warn.warn(str(x) + ") File not found: " + path)
            do_exist = False

    return do_exist