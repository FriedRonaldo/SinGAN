from tqdm import trange
from torch.nn import functional as F
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils
import pickle

from utils import *


def validateSinGAN(data_loader, networks, stage, args, additional=None):
    # set nets
    D = networks[0]
    G = networks[1]
    # switch to train mode
    D.eval()
    G.eval()
    # summary writer
    # writer = additional[0]
    val_it = iter(data_loader)

    z_rec = additional['z_rec']

    x_in = next(val_it)
    x_in = x_in.cuda(args.gpu, non_blocking=True)
    x_org = x_in

    x_in = F.interpolate(x_in, (args.size_list[stage], args.size_list[stage]), mode='bilinear', align_corners=True)
    vutils.save_image(x_in.detach().cpu(), os.path.join(args.res_dir, 'ORG_{}.png'.format(stage)),
                      nrow=1, normalize=True)
    x_in_list = [x_in]
    for xidx in range(1, stage + 1):
        x_tmp = F.interpolate(x_org, (args.size_list[xidx], args.size_list[xidx]), mode='bilinear', align_corners=True)
        x_in_list.append(x_tmp)

    for z_idx in range(len(z_rec)):
        z_rec[z_idx] = z_rec[z_idx].cuda(args.gpu, non_blocking=True)

    with torch.no_grad():
        x_rec_list = G(z_rec)

        # calculate rmse for each scale
        rmse_list = [1.0]
        for rmseidx in range(1, stage + 1):
            rmse = torch.sqrt(F.mse_loss(x_rec_list[rmseidx], x_in_list[rmseidx]))
            if args.validation:
                rmse /= 100.0
            rmse_list.append(rmse)
        if len(rmse_list) > 1:
            rmse_list[-1] = 0.0

        vutils.save_image(x_rec_list[-1].detach().cpu(), os.path.join(args.res_dir, 'REC_{}.png'.format(stage)),
                          nrow=1, normalize=True)

        for k in range(50):
            z_list = [F.pad(rmse_list[z_idx] * torch.randn(args.batch_size, 3, args.size_list[z_idx],
                                               args.size_list[z_idx]).cuda(args.gpu, non_blocking=True),
                            [5, 5, 5, 5], value=0) for z_idx in range(stage + 1)]
            x_fake_list = G(z_list)

            vutils.save_image(x_fake_list[-1].detach().cpu(), os.path.join(args.res_dir, 'GEN_{}_{}.png'.format(stage, k)),
                              nrow=1, normalize=True)



