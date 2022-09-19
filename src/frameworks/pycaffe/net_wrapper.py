from __future__ import print_function

# import cv2
import caffe
import os
import numpy as np
import warnings
import src.frameworks.pycaffe.tools as caffe_tools


class CaffeWrapper:
    """
    CaffeWrapper is object that handles nets using pycaffe interface.

    CaffeWrapper have the following properties:

    Attributes:
        model_root:     Directory containing model files (i.e, def and weight)
        model_def:      Filepath to prototext file (i.e., net architecture)
        model_weights:  Filepath of caffemodel file (i.e, weights)
        avg:            Mean img (to subtract from img during preprocessing)

    NOTE default configs are set to process VGG-Face-- currently not tested on other models (i.e., if attribute values
    were set to point at different pretrained CNN)

    TODO:
        Use transformer()
    """

    def __init__(self,  # note default assumes VGG model
                 model_root='',
                 model_def='models/vgg_face_caffe/VGG_FACE_deploy.prototxt',
                 model_weights='models/vgg_face_caffe/VGG_FACE.caffemodel',
                 avg=np.array([129.1863, 104.7624, 93.5940]),
                 k=2622,
                 mode='gpu',
                 net=None,
                 do_init=False,
                 gpu_id=0,
                 net_type='vgg',
                 transformer=None):

        """Instantiate Object."""
        """set members."""
        if not model_root:
            file_parts = os.path.split(model_def)
            self.model_root = f'{file_parts[0]}/'
            self.model_def = model_def
            self.model_weights = model_weights
        else:
            self.model_root = model_root
            self.model_def = self.model_root + model_def
            self.model_weights = self.model_root + model_weights

        self.avg = avg
        self.k = k

        self.net = net
        self.net_type = net_type
        self.mode = mode
        self.is_init = do_init

        self.gpu_id = gpu_id
        if do_init:
            self.init()

        # self.PsoOpts = pso_opts
        self.transformer = transformer

    def init(self):
        """
        Instantiates instance of net provided model def and weights
        :param net_type: network type (i.e., net reference) ['vgg' or 'res']
        :return:        None
        """

        print('Network type set', self.net_type)
        if self.mode == 'gpu':
            print("###### GPU ######")
            caffe.set_mode_gpu()
            caffe.set_device(self.gpu_id)
        else:
            print("****** CPU ******")
            caffe.set_mode_cpu()

        if not os.path.isfile(self.model_def):
            self.is_init = False
            warnings.warn(
                f'Caffe_Wrapper.init(): model definition file does not exit: {self.model_def}'
            )

            return

        if not os.path.isfile(self.model_weights):
            self.is_init = False
            warnings.warn(
                f'Caffe_Wrapper.init(): model weights file does not exit: {self.model_weights}'
            )

            return
        self.net = caffe.Net(self.model_def,  # defines structure of model
                             self.model_weights,  # contains the trained weights
                             caffe.TEST)  # test mode (e.g., no dropout)
        self.is_init = True

    def predict(self, img=None, f_image=''):
        """Return predictions for top N classes for img.

        Inputs:
            img:        image to classify

        Outputs:
            prob:    scores for top N class for img (= read(ifile))
        """
        # img += np.reshape(x,[224, 224])
        if not self.net:
            self.init()

        if img is None:
            caffe_tools.load_prepare_image_vgg(f_image=f_image)

        # self.net.forward_all( data = img )
        self.net.blobs['data'].data[...] = img
        return self.net.blobs['prob'].data[0]

    def extract_features(self, image, output_layer='fc7'):
        """
        Passes image through network, returning output of specified layer.

        :param image: image to process
        :param output_layer: layer of net to return output from [default 'prob']

        :return: net output wrt layer output_layer
        """

        # check whether net has been initialized
        if not self.is_init:
            self.init()
        if not self.is_init:
            return None

        self.net.blobs['data'].data[...] = image

        ### perform classification
        output = self.net.forward()
        # the output probability vector for the first image in the batch
        return self.net.blobs[output_layer].data.copy()

    def feed_forward(self, image, output_layer='prob'):
        """
        Passes image through network, returning output of specified layer.

        :param image: image to process
        :param output_layer: layer of net to return output from [default 'prob']

        :return: net output wrt layer output_layer
        """

        # check whether net has been initialized
        if not self.is_init:
            self.init()
        if not self.is_init:
            return None

        self.net.blobs['data'].data[...] = image

        ### perform classification

        output = self.net.forward()
        # the output probability vector for the first image in the batch
        return output[output_layer][0]

    def get_net_shape(self):
        return self.net.blobs['data'].shape

    def get_batch_size(self):
        return self.net.blobs['data'].shape[0]

    def set_batch_size(self, batch_size):
        new_shape = (batch_size,) + tuple(self.net.blobs['data'].shape[1:])
        self.net.blobs['data'].reshape(*new_shape)

    def set_transformer(self):
        # check whether net has been initialized
        if not self.is_init:
            self.init()
        if not self.is_init:
            # if still not initialized, then return
            return None

        transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        transformer.set_mean('data', self.avg)
        transformer.set_transpose('data', (2, 0, 1))

        transformer.set_raw_scale('data', 255.0)
        transformer.set_channel_swap('data', (2, 1, 0))
        self.transformer = transformer

    def get_transformer(self):
        # check whether net has been initialized
        if not self.transformer:
            print("Need to setup transformer")
            self.set_transformer()

        print("Returning transformer")
        return self.transformer

    ########################################################################################################################
    ###                                                                                                                  ###
    ###                                                    Layers                                                        ###
    ###                                                                                                                  ###
    ########################################################################################################################
    def print_layer_info(self):
        """
        Simple function that prints each layer in order (i.e., LAYER NAME : LAYER TYPE (No. BLOBS))
        :param net: Network to print info of
        :return: None
        """
        print("Network layers:")
        for name, layer in zip(self.net._layer_names, self.net.layers):
            print("{:<7}: {:17s}({} blobs)".format(name, layer.type, len(layer.blobs)))

    def get_layers(self):
        """
        Get the layer names of the network.

        :param net: caffe network
        :type net: caffe.Net
        :return: layer names
        :rtype: [string]
        """
        return list(self.net.params.keys())
        ########################################################################################################################
        ###                                                                                                                  ###
        ###                                                 Visualization                                                    ###
        ###                                                                                                                  ###
        ########################################################################################################################
        # def visualize_kernels(self, layer, zoom=5):
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
        #     assert layer in self.get_layers(), "layer %s not found" % layer
        #
        #     num_kernels = self.net.params[layer][0].data.shape[0]
        #     num_channels = self.net.params[layer][0].data.shape[1]
        #     kernel_height = self.net.params[layer][0].data.shape[2]
        #     kernel_width = self.net.params[layer][0].data.shape[3]
        #
        #     image = np.zeros((num_kernels * zoom * kernel_height, num_channels * zoom * kernel_width))
        #     for k in range(num_kernels):
        #         for c in range(num_channels):
        #             kernel = self.net.params[layer][0].data[k, c, :, :]
        #             kernel = cv2.resize(kernel, (zoom * kernel_height, zoom * kernel_width), kernel, 0, 0, cv2.INTER_NEAREST)
        #             kernel = (kernel - np.min(kernel)) / (np.max(kernel) - np.min(kernel))
        #             image[k * zoom * kernel_height:(k + 1) * zoom * kernel_height,
        #             c * zoom * kernel_width:(c + 1) * zoom * kernel_width] = kernel

        # return image

    # def visualize_weights(self, layer, zoom=2):
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
    #     assert layer in self.get_layers(), "layer %s not found" % layer
    #
    #     weights = self.net.params[layer][0].data
    #     weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
    #     return cv2.resize(weights, (weights.shape[0] * zoom, weights.shape[1] * zoom), weights, 0, 0, cv2.INTER_NEAREST)

    ########################################################################################################################
    ###                                                                                                                  ###
    ###                                                    Images                                                        ###
    ###                                                                                                                  ###
    ########################################################################################################################
    def load_image(self, f_image):
        """
        Function call loads image and formats as tensor in such to prepare for net_type (i.e., vgg vs stres)
        :param f_image: file path to image

        :return:    loaded tensor ready to feed through caffe.net()
        """
        if self.net_type == 'vgg':
            return caffe_tools.load_prepare_image_vgg(f_image)

    def caffe2RGB(self, f_image):
        """
        Function call loads image and formats as tensor in such to prepare for net_type (i.e., vgg vs stres)
        :param f_image: file path to image

        :return:    loaded tensor ready to feed through caffe.net()
        """
        if self.net_type == 'vgg':
            return caffe_tools.load_prepare_image_vgg(f_image)
