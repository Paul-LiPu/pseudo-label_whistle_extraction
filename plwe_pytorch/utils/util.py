import numpy as np
import torch
import torch.nn.init as init
import cv2
from utils.global_vars import dtype
import os
import math


class Config():
    def __init__(self):
        pass

def make_dir(dir):
    """
    If dir does not exist, create a folder at dir
    :param dir: the path to the target directory
    """
    if not os.path.exists(dir):
        os.makedirs(dir)

def removeLineFeed(line):
    """
    Remove the line feed character of the line
    :param line: string of one line
    :return: string without line feed
    """
    return line[:-1]

def read_file_list(file):
    """
    Read a list of filename from a txt file.
    Each file name is written in one line in the txt file.
    :param file: path to the txt file
    :return: a list of file name
    """
    with open(file) as f:
        data = f.readlines()
    data = list(map(removeLineFeed, data))
    return data


def weights_init_He_normal(m):
    """
    Initialize model params using He's method in https://arxiv.org/pdf/1502.01852.pdf
    :param m: pytorch module
    """
    classname = m.__class__.__name__
#     print classname
    if classname.find('Transpose') != -1:
        m.weight.data.normal_(0.0, 0.001)
        if not m.bias is None:
            m.bias.data.zero_()
    elif classname.find('Conv') != -1:
        # std = np.sqrt(2. / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels))
        # m.weight.data.normal_(0.0, std)
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        if not m.bias is None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.)
        if not m.bias is None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.001)
        if not m.bias is None:
            m.bias.data.zero_()


def cal_psnr(im1, im2):
    """
    calculate the PSNR metric of two images
    :param im1: image 1, numpy array with shape (h, w), dtype uint8
    :param im2: image 1, numpy array with shape (h, w), dtype uint8
    :return: PSNR
    """
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    if mse == 0:
        return -1
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


def evaluate_detection_network(net, test_dataset, config, iterations, test_batch_num, test_pic_num):
    """
    Evaluate prediction using PSNR, and write network output for target data specified by test_pic_num
    :param net: model object
    :param test_dataset: dataset object
    :param config: configuration objects
    :param iterations: number of training iterations before this evaluation
    :param test_batch_num: the mini-batch number for the target testing data
    :param test_pic_num: the index of the target testing data within a mini-batch
    :return: PSNR metric
    """
    psnr = []
    for i_test_batch in range(0, math.ceil(len(test_dataset) / config.test_batchsize)):
        test_batched = next(test_dataset)
        input = torch.from_numpy(np.asarray(test_batched[0])).type(dtype)
        output = net(input)
        output = np.clip((output.cpu().detach().numpy()) * 255., 0, 255).astype(np.uint8)
        label = np.clip(np.asarray(test_batched[1]) * 255., 0, 255).astype(np.uint8)

        if i_test_batch == test_batch_num:
            output_patch = test_pic_num
            output_image = np.clip(input.data.cpu().detach().numpy()[output_patch, 0, :, :] * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(config.test_folder + '/test_image_iter_' + str(iterations) + '_input.png', output_image)
            output_image = output[output_patch, 0, :, :]
            cv2.imwrite(config.test_folder + '/test_image_iter_'+str(iterations)+'_output.png', output_image)
            output_image = label[output_patch, 0, :, :]
            cv2.imwrite(config.test_folder + '/test_image_iter_' + str(iterations) + '_GT.png', output_image)
        for i in range(0, len(label)):
            test_psnr = cal_psnr(output[i,], label[i,])
            if test_psnr == -1:
                continue
            psnr.append(test_psnr)
    return psnr

def find_test_image_h5(test_dataset, config):
    """
    Find meaningful target data for testing
    :param test_dataset: dataset object
    :param config: configuration object
    :return:
        the mini-batch number for the target testing data
        the index of the target testing data within a mini-batch
    """
    for i_test_batch in range(0, len(test_dataset) // config.test_batchsize):
        test_batched = next(test_dataset)
        label = np.asarray(test_batched[1])
        label = np.reshape(label, (label.shape[0] * label.shape[1], label.shape[2] * label.shape[3]))
        label_sum = np.sum(label, axis=1)
        for i in range(label.shape[0]):
            if label_sum[i] > 0:
                test_dataset.curr_file = 0
                test_dataset.curr_file_pointer = 0
                return i_test_batch, np.argmax(label_sum)
