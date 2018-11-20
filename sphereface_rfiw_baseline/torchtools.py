import glob
import os
from collections import defaultdict

import PIL
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from common.io import is_file, file_base, mkdir

import shutil
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
ByteTensor = torch.cuda.ByteTensor if cuda else torch.ByteTensor
#
# if cuda:
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')


class TorchTools(object):

    @staticmethod
    def save_checkpoint(state, is_best=False, checkpoint_dir='./', filename='checkpoint.pth.tar'):
        """Save checkpoint if a new best is achieved"""
        torch.save(state, checkpoint_dir + filename)  # save checkpoint

        if is_best:
            print("=> Saving a new best")
            shutil.copy(checkpoint_dir + filename, checkpoint_dir + '/best_model.pth.tar')
        else:
            print("=> Validation Accuracy did not improve")


    @staticmethod
    def load_checkpoint(net, f_weights='/Users/joseph.robinson/Documents/logs/checkpoint.pth.tar'):
        if is_file(f_weights):
            print('Loading Checkpoint: ' + f_weights)
            if cuda:
                checkpoint = torch.load(f_weights)
            else:
                # Load GPU model on CPU
                checkpoint = torch.load(f_weights, map_location=lambda storage, loc: storage)
                net.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (trained for {} epochs)".format(f_weights, checkpoint['epoch']))

            return checkpoint['epoch'], checkpoint['best_acc']

    @staticmethod
    def adjust_learning_rate(optimizer, epoch, nepochs=30, stepsize=0.1, lr=0.001):
        """Sets learning rate to initial LR decayed by 'stepsize' (default=10) every 'nepochs' (defaults=30) epochs"""
        lr = lr * (0.1 ** (epoch // nepochs))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    @staticmethod
    def tensor2im(input_image, imtype=np.uint8):
        """
        Converts a Tensor into an image array (numpy)
        :param input_image:
        :param imtype:
        :return:
        :type:    the desired type of the converted numpy array
        """
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        return image_numpy.astype(imtype)

    @staticmethod
    def diagnose_network(net, name='network'):
        mean = 0.0
        count = 0
        for param in net.parameters():
            if param.grad is not None:
                mean += torch.mean(torch.abs(param.grad.data))
                count += 1
        if count > 0:
            mean = mean / count
        print(name)
        print(mean)

    @staticmethod
    def images2numpy(tensor):
        """
        convert pytorch tensor to numpy array
        :param tensor:
        :return:
        """
        try:
            generated = tensor.data.cpu().numpy().transpose(0, 2, 3, 1)
        except:
            generated = tensor.cpu().numpy().transpose(0, 2, 3, 1)

        generated[generated < -1] = -1
        generated[generated > 1] = 1
        generated = (generated + 1) / 2 * 255
        return generated.astype('uint8')

    @staticmethod
    def resume(model, epoch=None):
        """
        Resume training from checkpoint
        :param model:
        :param epoch:   specify epoch to resume from
        :return:
        """
        # if epoch is None:
        #     s = 'iter'
        #     epoch = 0
        # else:
        #     s = 'epoch'
        if epoch is None:
            return None

        index = 0 if epoch is not None else epoch
        return model.load_state_dict(torch.load(TorchTools.get_checkpoint_string(index, dir=model.checkdir)))
        # torch.load('checkpoint/encoder_{}_{:08d}.pth'.format(s, epoch)))

    @staticmethod
    def save(model, index, epoch=True):
        mkdir(model.checkdir)
        # s = 'epoch' if epoch else 'iter'
        torch.save(model.state_dict(), TorchTools.get_checkpoint_string(index, dir=model.checkdir, epoch=epoch))

    @staticmethod
    def count_parameters(model):
        n = 0
        for param in model.parameters():
            n += param.numel()
        return n

    @staticmethod
    def get_checkpoint_string(index, dir='checkpoint', unformatted_path='encoder_{}_{:08d}.pth', epoch=True):

        s = 'epoch' if epoch else 'iter'
        return os.path.join(dir, unformatted_path).format(s, index)

    @staticmethod
    def videos_to_numpy(tensor):
        generated = tensor.transpose(0, 1, 2, 3, 4)
        generated[generated < -1] = -1
        generated[generated > 1] = 1
        generated = (generated + 1) / 2 * 255
        return generated.astype('uint8')


    # @staticmethod
    # def debug_tensor(tensor, name):
    # """
    # Simple utility which helps with debugging.
    # Takes a tensor and outputs: min, max, avg, std, number of NaNs, number of
    # INFs.
    # :param tensor: torch tensor
    # :param name: name of the tensor (only for logging)
    # """
    # # mylog = Logger(arguments['--log_folder'])
    # # logging.info(name)
    # tensor = tensor.detach().float().cpu().numpy()
    # logging.info(f'MIN: {tensor.min()} MAX: {tensor.max()} '
    #              f'AVG: {tensor.mean()} STD: {tensor.std()} '
    #              f'NAN: {np.isnan(tensor).sum()} INF: {np.isinf(tensor).sum()}')
