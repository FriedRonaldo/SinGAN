from tqdm import trange
from torch.nn import functional as F
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utils import *
from ops import compute_grad_gp_wgan, compute_grad_gp
import torchvision.utils as vutils


def trainSinGAN(data_loader, networks, opts, stage, args, additional):
    # avg meter
    d_losses = AverageMeter()
    g_losses = AverageMeter()

    # set nets
    D = networks[0]
    G = networks[1]
    # set opts
    d_opt = opts['d_opt']
    g_opt = opts['g_opt']
    # switch to train mode
    D.train()
    G.train()
    # summary writer
    # writer = additional[0]
    train_it = iter(data_loader)
    # total_iter = 2000 * (args.num_scale - stage + 1)
    # decay_lr = 1600 * (args.num_scale - stage + 1)
    total_iter = 2000
    decay_lr = 1600

    d_iter = 3
    g_iter = 3

    t_train = trange(0, total_iter, initial=0, total=total_iter)

    z_rec = additional['z_rec']

    for z_idx in range(len(z_rec)):
        z_rec[z_idx] = z_rec[z_idx].cuda(args.gpu, non_blocking=True)

    x_in = next(train_it)

    x_in = x_in.cuda(args.gpu, non_blocking=True)
    x_org = x_in
    x_in = F.interpolate(x_in, (args.size_list[stage], args.size_list[stage]), mode='bilinear', align_corners=True)
    vutils.save_image(x_in.detach().cpu(), os.path.join(args.res_dir, 'ORGTRAIN_{}.png'.format(stage)),
                      nrow=1, normalize=True)

    x_in_list = [x_in]
    for xidx in range(1, stage + 1):
        x_tmp = F.interpolate(x_org, (args.size_list[xidx], args.size_list[xidx]), mode='bilinear', align_corners=True)
        x_in_list.append(x_tmp)

    for i in t_train:
        if i == decay_lr:
            for param_group in d_opt.param_groups:
                    param_group['lr'] *= 0.1
                    print("DISCRIMINATOR LEARNING RATE UPDATE TO :", param_group['lr'])
            for param_group in g_opt.param_groups:
                    param_group['lr'] *= 0.1
                    print("GENERATOR LEARNING RATE UPDATE TO :", param_group['lr'])

        for _ in range(g_iter):
            g_opt.zero_grad()

            x_rec_list = G(z_rec)

            g_rec = F.mse_loss(x_rec_list[-1], x_in)
            # calculate rmse for each scale
            rmse_list = [1.0]
            for rmseidx in range(1, stage + 1):
                rmse = torch.sqrt(F.mse_loss(x_rec_list[rmseidx], x_in_list[rmseidx]))
                rmse_list.append(rmse)

            z_list = [F.pad(rmse_list[z_idx] * torch.randn(args.batch_size, 3, args.size_list[z_idx],
                                               args.size_list[z_idx]).cuda(args.gpu, non_blocking=True),
                            [5, 5, 5, 5], value=0) for z_idx in range(stage + 1)]

            x_fake_list = G(z_list)

            g_fake_logit = D(x_fake_list[-1])

            ones = torch.ones_like(g_fake_logit).cuda(args.gpu)

            if args.gantype == 'wgangp':
                # wgan gp
                g_fake = -torch.mean(g_fake_logit, (2, 3))
                g_loss = g_fake + 10.0 * g_rec
            elif args.gantype == 'zerogp':
                # zero centered GP
                g_fake = F.binary_cross_entropy_with_logits(g_fake_logit, ones, reduction='none').mean()
                g_loss = g_fake + 100.0 * g_rec

            elif args.gantype == 'lsgan':
                # lsgan
                g_fake = F.mse_loss(torch.mean(g_fake_logit, (2, 3)), 0.9 * ones)
                g_loss = g_fake + 50.0 * g_rec

            g_loss.backward()
            g_opt.step()

            g_losses.update(g_loss.item(), x_in.size(0))

        # Update discriminator
        for _ in range(d_iter):
            x_in.requires_grad = True

            d_opt.zero_grad()
            x_fake_list = G(z_list)

            d_fake_logit = D(x_fake_list[-1].detach())
            d_real_logit = D(x_in)

            ones = torch.ones_like(d_real_logit).cuda(args.gpu)
            zeros = torch.zeros_like(d_fake_logit).cuda(args.gpu)

            if args.gantype == 'wgangp':
                # wgan gp
                d_fake = torch.mean(d_fake_logit, (2, 3))
                d_real = -torch.mean(d_real_logit, (2, 3))
                d_gp = compute_grad_gp_wgan(D, x_in, x_fake_list[-1], args.gpu)
                d_loss = d_real + d_fake + 0.1 * d_gp
            elif args.gantype == 'zerogp':
                # zero centered GP
                # d_fake = F.binary_cross_entropy_with_logits(torch.mean(d_fake_logit, (2, 3)), zeros)
                d_fake = F.binary_cross_entropy_with_logits(d_fake_logit, zeros, reduction='none').mean()
                # d_real = F.binary_cross_entropy_with_logits(torch.mean(d_real_logit, (2, 3)), ones)
                d_real = F.binary_cross_entropy_with_logits(d_real_logit, ones, reduction='none').mean()
                d_gp = compute_grad_gp(torch.mean(d_real_logit, (2, 3)), x_in)
                d_loss = d_real + d_fake + 10.0 * d_gp

            elif args.gantype == 'lsgan':
                # lsgan
                d_fake = F.mse_loss(torch.mean(d_fake_logit, (2, 3)), zeros)
                d_real = F.mse_loss(torch.mean(d_real_logit, (2, 3)), 0.9 * ones)
                d_loss = d_real + d_fake

            d_loss.backward()
            d_opt.step()

            d_losses.update(d_loss.item(), x_in.size(0))

        t_train.set_description('Stage: [{}/{}] Avg Loss: D[{d_losses.avg:.3f}] G[{g_losses.avg:.3f}] RMSE[{rmse:.3f}]'
                                .format(stage, args.num_scale, d_losses=d_losses, g_losses=g_losses, rmse=rmse_list[-1]))
