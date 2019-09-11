from __future__ import print_function


import os
import matplotlib.pyplot as plt
import numpy as np
import io
from configparser import ConfigParser, ExtendedInterpolation
from collections import namedtuple
import warnings as warn

__author__ = 'Joseph Robinson'


def is_linux():
    """is the current platform Linux?"""
    (sysname, nodename, release, version, machine) = os.uname()
    return sysname == 'Linux'


def is_macosx():
    """Is the current platform MacOSX?"""
    (sysname, nodename, release, version, machine) = os.uname()
    return sysname == 'Darwin'


def is_number(obj):
    """Is a python object a number?"""
    try:
        complex(obj)  # for int, long, float and complex
    except ValueError:
        return False

    return True


def is_numpy(obj):
    """Is a python object a numpy array?"""
    return 'numpy' in str(type(obj))


def parse(cfile=None):
    """
    Instantiates parser for INI (config) file
    :param cfile: absolute filepath to config INI file

    :return: ConfigParser object with configurations loaded
    """
    if not cfile:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        cfile = os.path.join(dir_path, 'my_bb_configs.ini')

    print('Loading configs: ' + cfile)
    parser = ConfigParser(interpolation=ExtendedInterpolation())
    parser.read(cfile)

    return parser
# end parse()


def print_configs(opts, header=''):
    """
    Simple function that prints configurations
    :param opts: configs of certain type (named tuple)
    :param header: Title (reference) to display above printout
    """
    if header:
        print('\n###############################################################\n')
        print('\t########\t {} \t########\n'.format(header))
        print('###############################################################\n')

    for field in opts._fields:
        if len(field) < 8:
            print('\t{}\t\t\t:\t{}\n'.format(field, getattr(opts, field)))
        else:
            print('\t{}\t\t:\t{}\n'.format(field, getattr(opts, field)))


def show(img_display, img, lmarks, frontal_raw, face_proj, background_proj, temp_proj2_out_2, sym_weight):
    plt.ion()
    plt.show()
    plt.subplot(221)
    plt.title('Query Image')
    plt.imshow(img_display[:, :, ::-1])
    plt.axis('off')

    plt.subplot(222)
    plt.title('Landmarks Detected')
    plt.imshow(img[:, :, ::-1])
    plt.scatter(lmarks[0][:, 0], lmarks[0][:, 1], c='red', marker='.', s=100, alpha=0.5)
    plt.axis('off')
    plt.subplot(223)
    plt.title('Rendering')

    plt.imshow(frontal_raw[:, :, ::-1])
    plt.axis('off')

    plt.subplot(224)
    if sym_weight is None:
        plt.title('Face Mesh Projected')
        plt.imshow(img[:, :, ::-1])
        plt.axis('off')
        face_proj = np.transpose(face_proj)
        plt.plot(face_proj[1:-1:100, 0], face_proj[1:-1:100, 1], 'b.')
        background_proj = np.transpose(background_proj)
        temp_proj2_out_2 = temp_proj2_out_2.T
        plt.plot(background_proj[1:-1:100, 0], background_proj[1:-1:100, 1], 'r.')
        plt.plot(temp_proj2_out_2[1:-1:100, 0], temp_proj2_out_2[1:-1:100, 1], 'm.')
    else:
        plt.title('Face Symmetry')
        plt.imshow(sym_weight)
        plt.axis('off')
        plt.colorbar()

    plt.draw()
    plt.pause(0.001)
    __ = input("Press [enter] to continue.")
    plt.clf()



class Configurations:
    """
    Simple class made-up of static methods for handling INI configuration files used in LOKI API.
    """
    header = "Configurations"

    @staticmethod
    def parse(configs_file):
        """
        Instantiates parser for INI (config) file
        :param cfconfigs_fileile: absolute path to config INI file

        :return: ConfigParser object with configurations loaded
        """
        if not configs_file:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            configs_file = os.path.join(dir_path, 'my_bb_configs.ini')

        print('Loading configs: ' + configs_file)
        # parser = ConfigParser(interpolation=ExtendedInterpolation())
        # parser.read(configs_file)

        # return parser

    @staticmethod
    def get_config(f_configs, section=None):
        """ Top-level function that reads & stores all (or specified sections of) configurations stored as INI files.

            Named-tuples are used to store all values of configs, with fields named as parameter (variables) of INI.

                :param f_configs: configs filepath (INI)
                :param section: typename of namedtuple object

                :return namedtuple with configs set as proper type
        """
        if io.is_file(f_configs) is False:
            warn.warn('configs (INI) file does not exist: ' + f_configs)
            return None

    @staticmethod
    def print_configs(configs, header=''):
        """
        Simple function that prints configurations
        :param opts: configs of certain type (named tuple)
        :param header: Title (reference) to display above printout
        """
        if header:
            print('\n###############################################################\n')
            print('\t########\t {} \t########\n'.format(header))
            print('###############################################################\n')
        for fld in configs._fields:
            if len(fld) < 8:
                print('\t{}\t\t\t:\t{}\n'.format(fld, getattr(configs, fld)))
            else:
                print('\t{}\t\t:\t{}\n'.format(fld, getattr(configs, fld)))

    @staticmethod
    def get_caffe_configs(f_configs, header='NetOpts'):
        """
            Get configurations for a specific section of INI file.

            Default: function loads and returns configs for section [pso]
        """

        parser = Configurations.parse(f_configs)
        # Parse general options
        dic = dict(parser.items('net'))

        NetOpts = namedtuple(header, dic.keys())

        NetOpts.idim = int(dic['idim'])
        NetOpts.model_def = str(dic['model_def'])


        NetOpts.model_weights = str(dic['model_weights'])
        NetOpts.mode = str(dic['mode'])
        NetOpts.gpu_id = int(dic['gpu_id'])
        NetOpts.lfile = str(dic['lfile'])

        NetOpts.k = int(dic['k'])

        mdims = dic['mdims']
        mdims = mdims.replace('[', '').replace(']', '').split(',')

        NetOpts.mdims = np.array(
            [int(mdims[0]), int(mdims[1]), int(mdims[2]), int(mdims[3])]
        )

        sp_avg = dic['avg'].replace('[', '').replace(']', '').split(',')
        if len(sp_avg) == 3:
            NetOpts.avg = np.array([float(sp_avg[0]), float(sp_avg[1]), float(sp_avg[2])])
        else:
            NetOpts.avg = float(sp_avg)


        NetOpts.model_root = str(dic['model_root'])

        return NetOpts


