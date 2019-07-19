import numpy as np
from glob import glob
import os
from datasets.datasetgetter import get_dataset
import argparse
import torchvision.utils as vutils
import torch.nn.functional as F
import torch
from models.generator import Generator
from models.discriminator import Discriminator

parser = argparse.ArgumentParser(description='PyTorch Simultaneous Training')
parser.add_argument('--data_dir', default='../data/', help='path to dataset')
parser.add_argument('--model_name', type=str, default='SinGAN', help='model name')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=80, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Total batch size - e.g) num_gpus = 2 , batch_size = 128 then, effectively, 64')
parser.add_argument('--val_batch', default=1, type=int)
parser.add_argument('--img_size_max', default=250, type=int, help='Input image size')
parser.add_argument('--img_size_min', default=25, type=int, help='Input image size')
parser.add_argument('--log_step', default=50, type=int, help='print frequency (default: 50)')
parser.add_argument('--load_model', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: None)')
parser.add_argument('--validation', dest='validation', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--test', dest='test', action='store_true',
                    help='test model on validation set')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--gpu', default=None, type=str,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--port', default='8888', type=str)


args = parser.parse_args()

# large = 250
# small = 25
#
# scale = large / small
#
# # scale = (4/3)^N / N = log_{4/3}(scale) = log(scale)/log(4/3)
#
# num_scale = int(np.round(np.log(scale) / np.log(4/3)))
#
# N = int(np.log(scale) / np.log(4/3))
#
# train_, _ = get_dataset('photo', args)
#
# trainiter = iter(train_)
#
# x = next(trainiter)
#
# x = x.unsqueeze(0)
#
#
# size_list = [int(args.img_size_min * (4/3)**i) for i in range(num_scale + 1)]
# print(size_list)
# G = Generator(args.img_size_min, num_scale)
# D = Discriminator()
#
# z_list = [F.pad(torch.randn(args.batch_size, 3, size_list[i], size_list[i]), [5, 5, 5, 5], value=-1) for i in range(num_scale + 1)]
#
# print('latent vector sizes')
# for z in z_list:
#     print(z.shape)
# print('-------------------')
#
# for i in range(num_scale + 1):
#     print('output sizes')
#     out = G(z_list)
#     for o in out:
#         print(o.shape)
#     d_out = torch.mean(D(out[-1]), (2, 3))
#     print(d_out.shape)
#     G.progress()
#     D.progress()
#     print(G.current_scale)
#     print(D.current_scale)
#     print('-------------------')
#
# print('-------------------')
#
# for key, val in G.sub_generators[0].named_parameters():
#     val.requires_grad = False
#
# print('-------------------')
#
# for i in range(len(out)):
#     vutils.save_image(out[i], 'tmp{}.png'.format(i), 1, normalize=True)
#

x1 = torch.rand(1, 1, 1, 1)

print(x1)