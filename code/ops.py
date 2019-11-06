from torch import autograd
from torch import nn
from torch.nn import functional as F
import math
from scipy import signal
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

# from inception import InceptionV3


def compute_grad_gp(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def compute_grad_gp_wgan(D, x_real, x_fake, gpu):
    alpha = torch.rand(x_real.size(0), 1, 1, 1).cuda(gpu)

    x_interpolate = ((1 - alpha) * x_real + alpha * x_fake).detach()
    x_interpolate.requires_grad = True
    d_inter_logit = D(x_interpolate)
    grad = torch.autograd.grad(d_inter_logit, x_interpolate,
                               grad_outputs=torch.ones_like(d_inter_logit), create_graph=True)[0]

    norm = grad.view(grad.size(0), -1).norm(p=2, dim=1)

    d_gp = ((norm - 1) ** 2).mean()
    return d_gp


def conv_cond_concat(x, y):
    if y.device == torch.device('cpu'):
        return torch.cat((x, y * torch.ones(x.shape[0], y.shape[1], x.shape[2], x.shape[3])), 1)
    else:
        return torch.cat((x, y * torch.ones(x.shape[0], y.shape[1], x.shape[2], x.shape[3]).cuda(y.device, non_blocking=True)), 1)


def generate_y_rand(size):
    # bang  black   blond   brown   gray    male    mustache     smile   glasses young
    # 12/60 15/60   10/60   12/60   6/60    30/60   6/60         30/60   6/60    48/60
    total_size = size
    tmp_rand = []
    tmp_rand.append(np.random.choice(2, total_size, p=[48. / 60., 12. / 60.]))
    tmp = np.random.choice(5, total_size, p=[17. / 60., 15. / 60., 10. / 60., 12. / 60., 6. / 60.])
    tmp_z = np.zeros([total_size, 4])
    i = 0
    for m in tmp:
        if m != 0:
            tmp_z[i][m - 1] = 1
        i += 1

    tmp_z = tmp_z.transpose()
    for z in tmp_z:
        tmp_rand.append(z)
    tmp_rand.append(np.random.choice(2, total_size, p=[30. / 60., 30. / 60.]))
    tmp_rand.append(np.random.choice(2, total_size, p=[54. / 60., 6. / 60.]))
    tmp_rand.append(np.random.choice(2, total_size, p=[30. / 60., 30. / 60.]))
    tmp_rand.append(np.random.choice(2, total_size, p=[54. / 60., 6. / 60.]))
    tmp_rand.append(np.random.choice(2, total_size, p=[12. / 60., 48. / 60.]))

    y_rand_total = np.asarray(tmp_rand).transpose()
    return y_rand_total.astype(np.float32)


class CInstanceNorm(nn.Module):
    def __init__(self, nfilter, nlabels):
        super().__init__()
        # Attributes
        self.nlabels = nlabels
        self.nfilter = nfilter
        # Submodules
        self.alpha_embedding = nn.Embedding(nlabels, nfilter)
        self.beta_embedding = nn.Embedding(nlabels, nfilter)
        self.bn = nn.InstanceNorm2d(nfilter, affine=False)
        # Initialize
        nn.init.uniform(self.alpha_embedding.weight, -1., 1.)
        nn.init.constant_(self.beta_embedding.weight, 0.)

    def forward(self, x, y):
        dim = len(x.size())
        batch_size = x.size(0)
        assert(dim >= 2)
        assert(x.size(1) == self.nfilter)

        s = [batch_size, self.nfilter] + [1] * (dim - 2)
        alpha = self.alpha_embedding(y)
        alpha = alpha.view(s)
        beta = self.beta_embedding(y)
        beta = beta.view(s)

        out = self.bn(x)
        out = alpha * out + beta

        return out


def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def update_average(model_tgt, model_src, beta=0.999):
    # model_tgt : deep copy of generator (used for test)
    toogle_grad(model_src, False)
    toogle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False, momentum=momentum)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class MultiConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False, momentum=momentum)
        self.gamma = nn.Linear(num_classes, num_features)
        self.beta = nn.Linear(num_classes, num_features)

    def forward(self, x, y):
        out = self.bn(x)
        gamma = self.gamma(y)
        beta = self.beta(y)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class SpatialAdaptiveNorm2d(nn.Module):
    def __init__(self, num_features, hid_features=64, momentum=0.1, num_classes=0):
        super().__init__()
        self.num_features = num_features
        self.hid_features = hid_features
        self.num_classes = num_classes
        if num_classes > 0:
            self.bn = MultiConditionalBatchNorm2d(num_features, num_classes)
        else:
            self.bn = nn.BatchNorm2d(num_features, affine=False, momentum=momentum)

        # Not apply SN to SPADE (original code)
        self.hidden = nn.Sequential(nn.Conv2d(3, hid_features, 3, 1, 1),
                                    nn.LeakyReLU(2e-1))

        self.gamma = nn.Conv2d(hid_features, num_features, 3, 1, 1)
        self.beta = nn.Conv2d(hid_features, num_features, 3, 1, 1)
        # make one more hidden for z and concat to out then, get gamma and beta
        # or resize z * mask + img then, get hid -> gamma, beta

    def forward(self, feat, img, y=None):
        rimg = F.interpolate(img, size=feat.size()[2:])
        if self.num_classes > 0 and y is not None:
            feat = self.bn(feat, y)
        else:
            feat = self.bn(feat)
        out = self.hidden(rimg)
        gamma = self.gamma(out)
        beta = self.beta(out)
        out = gamma * feat + beta
        return out


class SpatialModulatedNorm2d(nn.Module):
    def __init__(self, num_features, hid_features=64, momentum=0.1, num_classes=0):
        super().__init__()
        self.num_features = num_features
        self.hid_features = hid_features
        self.num_classes = num_classes
        if num_classes > 0:
            self.bn = MultiConditionalBatchNorm2d(num_features, num_classes, momentum=momentum)
        else:
            self.bn = nn.BatchNorm2d(num_features, affine=False, momentum=momentum)

        self.hidden_img = nn.Sequential(nn.Conv2d(3 + 16, hid_features, 3, 1, 1),
                                        nn.LeakyReLU(2e-1))

        # self.hidden_z = nn.Sequential(nn.Conv2d(16, hid_features//2, 3, 1, 1),
        #                                 nn.LeakyReLU(2e-1))

        self.gamma = nn.Conv2d(hid_features, num_features, 3, 1, 1)
        self.beta = nn.Conv2d(hid_features, num_features, 3, 1, 1)
        # make one more hidden for z and concat to out then, get gamma and beta
        # or resize z * mask + img then, get hid -> gamma, beta

    def forward(self, feat, img, z, y=None):
        rimg = F.interpolate(img, size=feat.size()[2:])
        rz = F.interpolate(z, size=feat.size()[2:])

        rin = torch.cat((rimg, rz), 1)
        out = self.hidden_img(rin)

        if self.num_classes > 0 and y is not None:
            feat = self.bn(feat, y)
        else:
            feat = self.bn(feat)

        # out = torch.cat((hidimg, hidz), 1)
        gamma = self.gamma(out)
        beta = self.beta(out)
        out = gamma * feat + beta

        return out
# class SpatialModulatedNorm2d(nn.Module):
#     def __init__(self, num_features, hid_features=64, momentum=0.1, num_classes=0):
#         super().__init__()
#         self.num_features = num_features
#         self.hid_features = hid_features
#         self.num_classes = num_classes
#         if num_classes > 0:
#             # self.bn = MultiConditionalBatchNorm2d(num_features, num_classes, momentum=momentum)
#             self.bn = SelfModulratedBatchNorm2d(num_features, num_latent=256, num_classes=num_classes, momentum=momentum)
#         else:
#             self.bn = SelfModulratedBatchNorm2d(num_features, num_latent=256, momentum=momentum)
#
#         self.hidden_img = nn.Sequential(nn.Conv2d(3, hid_features, 3, 1, 1),
#                                     nn.LeakyReLU(2e-1))
#
#         self.gamma = nn.Conv2d(hid_features, num_features, 3, 1, 1)
#         self.beta = nn.Conv2d(hid_features, num_features, 3, 1, 1)
#         # make one more hidden for z and concat to out then, get gamma and beta
#         # or resize z * mask + img then, get hid -> gamma, beta
#
#     def forward(self, feat, img, z, y=None):
#         rimg = F.interpolate(img, size=feat.size()[2:])
#
#         hidimg = self.hidden_img(rimg)
#
#         if self.num_classes > 0 and y is not None:
#             feat = self.bn(feat, z.view(z.size(0), -1), y)
#         else:
#             feat = self.bn(feat, z.view(z.size(0), -1))
#
#         gamma = self.gamma(hidimg)
#         beta = self.beta(hidimg)
#         out = gamma * feat + beta
#
#         return out


class SelfModulratedBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_latent, num_hidden=0, num_classes=0, momentum=0.1):
        super(SelfModulratedBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.num_latent = num_latent
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.bn = nn.BatchNorm2d(num_features, affine=False, momentum=momentum)
        if num_hidden > 0:
            self.fc_z = nn.Sequential(nn.Linear(num_latent, num_hidden), nn.ReLU(True))
            num_latent = num_hidden
        self.gamma = nn.Linear(num_latent, num_features)
        self.beta = nn.Linear(num_latent, num_features)
        if num_classes > 0:
            self.fc_y1 = nn.Linear(num_classes, num_latent)
            self.fc_y2 = nn.Linear(num_classes, num_latent)

    def forward(self, h, z, y=None):
        if self.num_hidden > 0:
            z = self.fc_z(z)
        if y is not None and self.num_classes > 0:
            z = z + self.fc_y1(y) + z * self.fc_y2(y)

        out = self.bn(h)
        gamma = self.gamma(z)
        beta = self.beta(z)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class Embedding(torch.nn.Embedding):
    def __init__(self, *args, spectral_norm_pi=1, **kwargs):
        super(Embedding, self).__init__(*args, **kwargs)
        self.spectral_norm_pi = spectral_norm_pi
        if spectral_norm_pi > 0:
            self.register_buffer("u", torch.randn((1, self.num_embeddings), requires_grad=False))
        else:
            self.register_buffer("u", None)

    def forward(self, input):
        if self.spectral_norm_pi > 0:
            w_mat = self.weight.view(self.num_embeddings, -1)
            u, sigma, _ = max_singular_value(w_mat, self.u, self.spectral_norm_pi)
            w_bar = torch.div(self.weight, sigma)
            if self.training:
                self.u = u
        else:
            w_bar = self.weight

        return F.embedding(
            input, w_bar, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)


def max_singular_value(w_mat, u, power_iterations):

    for _ in range(power_iterations):
        v = l2normalize(torch.mm(u, w_mat.data))

        u = l2normalize(torch.mm(v, torch.t(w_mat.data)))

    sigma = torch.sum(torch.mm(u, w_mat) * v)

    return u, sigma, v


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class GCRNNCellBase(torch.nn.Module):

    def __init__(self, latent_size, ch, bias, num_hidden=3, sn=False):
        super(GCRNNCellBase, self).__init__()
        self.ch = ch
        self.latent_size = latent_size
        self.concat_size = self.latent_size//4

        if sn:
            # squeeze ... not necessary? ( concat_size -> latent_size ? )
            self.layer_zh = torch.nn.utils.spectral_norm(torch.nn.Conv2d(latent_size, self.concat_size, 3, 1, 1, bias=bias))

            self.layer_hh = torch.nn.ModuleList([torch.nn.utils.spectral_norm(torch.nn.Conv2d(ch, self.concat_size, 3, 1, 1, bias=bias))])

            nf = 2 * self.concat_size

            for i in range(num_hidden - 1):
                self.layer_hh.append(torch.nn.Sequential(torch.nn.utils.spectral_norm(torch.nn.Conv2d(nf, 2 * nf, 3, 1, 1, bias=bias)),
                                                         torch.nn.BatchNorm2d(2 * nf),
                                                         torch.nn.ReLU(True)))
                nf *= 2

            self.layer_hh.append(torch.nn.Sequential(torch.nn.utils.spectral_norm(torch.nn.Conv2d(nf, 3, 3, 1, 1, bias=bias)),
                                                     torch.nn.Tanh()))
        else:
            # squeeze ... unnecessary? ( concat_size -> latent_size ? )
            self.layer_zh = torch.nn.Conv2d(latent_size, self.concat_size, 3, 1, 1, bias=bias)

            self.layer_hh = torch.nn.ModuleList([torch.nn.Conv2d(ch, self.concat_size, 3, 1, 1, bias=bias)])

            nf = 2 * self.concat_size

            for i in range(num_hidden - 1):
                self.layer_hh.append(torch.nn.Sequential(torch.nn.Conv2d(nf, 2 * nf, 3, 1, 1, bias=bias),
                                                         torch.nn.BatchNorm2d(2 * nf),
                                                         torch.nn.ReLU(True)))
                nf *= 2

            self.layer_hh.append(torch.nn.Sequential(torch.nn.Conv2d(nf, 3, 3, 1, 1, bias=bias),
                                                     torch.nn.Tanh()))

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.latent_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.latent_size))


class GCRNNCell(GCRNNCellBase):

    def __init__(self, latent_size, ch=3, bias=True, num_hidden=3, base=4, sn=False):
        super(GCRNNCell, self).__init__(latent_size, ch, bias, num_hidden=num_hidden, sn=sn)
        self.base = base

    def forward(self, z, hx=None, is_up=True):
        self.check_forward_input(z)
        if hx is None:
            hx = torch.ones(z.size(0), 3, self.base, self.base, requires_grad=False)
            if z.is_cuda:
                hx = hx.cuda(z.device, non_blocking=True)

        # print(z.device)
        # with torch.cuda.device_of(z.data):
        z = z.repeat(1, 1, hx.size(2), hx.size(3))
        # print(z.device)
        z_cat = self.layer_zh(z)
        h_cat = self.layer_hh[0](hx)
        # z_cat = z_cat.expand_as(h_cat)

        h_out = torch.cat((h_cat, z_cat), 1)

        for block in self.layer_hh[1:-1]:
            h_out = block(h_out)

        if is_up:
            h_out = F.interpolate(h_out, scale_factor=2)

        h_out = self.layer_hh[-1](h_out)

        return h_out


class DCRNNCellBase(torch.nn.Module):

    def __init__(self, nf, ch, bias, num_hidden=3, sn=False):
        super(DCRNNCellBase, self).__init__()
        self.ch = ch
        self.nf = nf

        if sn:
            self.layer_xhh = torch.nn.ModuleList(
                [torch.nn.Sequential(torch.nn.utils.spectral_norm(torch.nn.Conv2d(ch + self.nf, self.nf, 3, 1, 1, bias=bias)),
                                     torch.nn.LeakyReLU(inplace=True))])

            nf_ = self.nf

            for i in range(num_hidden - 1):
                self.layer_xhh.append(torch.nn.Sequential(torch.nn.utils.spectral_norm(torch.nn.Conv2d(nf_, 2 * nf_, 3, 1, 1, bias=bias)),
                                                          torch.nn.BatchNorm2d(2 * nf_),
                                                          torch.nn.LeakyReLU(True)))
                nf_ *= 2

            self.layer_xhh.append(torch.nn.Sequential(torch.nn.utils.spectral_norm(torch.nn.Conv2d(nf_, self.nf, 3, 1, 1, bias=bias)),
                                                      torch.nn.BatchNorm2d(self.nf),
                                                      torch.nn.LeakyReLU()))
        else:
            self.layer_xhh = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Conv2d(ch + self.nf, self.nf, 3, 1, 1, bias=bias),
                                                                      torch.nn.LeakyReLU(inplace=True))])

            nf_ = self.nf

            for i in range(num_hidden - 1):
                self.layer_xhh.append(torch.nn.Sequential(torch.nn.Conv2d(nf_, 2 * nf_, 3, 1, 1, bias=bias),
                                                          torch.nn.BatchNorm2d(2 * nf_),
                                                          torch.nn.LeakyReLU(True)))
                nf_ *= 2

            self.layer_xhh.append(torch.nn.Sequential(torch.nn.Conv2d(nf_, self.nf, 3, 1, 1, bias=bias),
                                                      torch.nn.BatchNorm2d(self.nf),
                                                      torch.nn.LeakyReLU()))

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.latent_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.latent_size))


class DCRNNCell(DCRNNCellBase):

    def __init__(self, nf=64, ch=3, bias=True, num_hidden=3, sn=False):
        super(DCRNNCell, self).__init__(nf, ch, bias, num_hidden=num_hidden, sn=sn)

    def forward(self, x, hx=None, is_down=True):
        # self.check_forward_input(x)
        if hx is None:
            hx = torch.ones(x.size(0), self.nf, x.size(2), x.size(3), requires_grad=False)
            if x.is_cuda:
                hx = hx.cuda(x.device, non_blocking=True)

        h_out = torch.cat((x, hx), 1)

        for block in self.layer_xhh[:-1]:
            h_out = block(h_out)

        if is_down:
            h_out = F.interpolate(h_out, scale_factor=0.5)

        h_out = self.layer_xhh[-1](h_out)

        return h_out


def mixup_criterion(x_orig, x_flip, x_recon, lam, loss='l1'):
    if loss == 'l1':
        return lam * F.l1_loss(x_recon, x_orig) + (1 - lam) * F.l1_loss(x_recon, x_flip)
    else:
        return lam * F.mse_loss(x_recon, x_orig) + (1 - lam) * F.mse_loss(x_recon, x_flip)


def eval_ssim(img1, img2, max_val=255, filter_size=11,
                       filter_sigma=1.5, k1=0.01, k2=0.03):
    """Return the Structural Similarity Map between `img1` and `img2`.

    This function attempts to match the functionality of ssim_index_new.m by
    Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

    Arguments:
      img1: Numpy array holding the first RGB image batch.
      img2: Numpy array holding the second RGB image batch.
      max_val: the dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
      filter_size: Size of blur kernel to use (will be reduced for small
      images).
      filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
        for small images).
      k1: Constant used to maintain stability in the SSIM calculation (0.01 in
        the original paper).
      k2: Constant used to maintain stability in the SSIM calculation (0.03 in
        the original paper).

    Returns:
      Pair containing the mean SSIM and contrast sensitivity between `img1` and
      `img2`.

    Raises:
      RuntimeError: If input images don't have the same shape or don't have four
        dimensions: [batch_size, height, width, depth].
    """
    if img1.shape != img2.shape:
        raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                           img1.shape, img2.shape)
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d',
                           img1.ndim)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    _, height, width, _ = img1.shape

    # Filter size can't be larger than height or width of images.
    size = min(filter_size, height, width)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
        window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
        mu1 = signal.fftconvolve(img1, window, mode='valid')
        mu2 = signal.fftconvolve(img2, window, mode='valid')
        sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
        sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
        sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
    cs = np.mean(v1 / v2)
    return ssim, cs


def eval_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def _FSpecialGauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


# def calc_generator_fid(model, data_loader, args, dims=2048):
#     eps = 1e-6
#
#     incept = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[dims]])
#
#     model.eval()
#     incept.eval()
#
#     valiter = iter(data_loader)
#
#     tot_iter = len(valiter)
#
#     pred_fake = np.empty((args.val_batch * tot_iter, dims))
#     pred_real = np.empty((args.val_batch * tot_iter, dims))
#
#     if not next(model.parameters()).device == torch.device('cpu'):
#         incept = incept.cuda(args.gpu)
#
#     for i in tqdm(range(tot_iter)):
#         x_real, _ = next(valiter)
#         z_in = torch.randn(args.val_batch, args.latent_size)
#         if not next(model.parameters()).device == torch.device('cpu'):
#             x_real = x_real.cuda(args.gpu, non_blocking=True)
#             z_in = z_in.cuda(args.gpu, non_blocking=True)
#         out = model(z_in)
#         x_fake = out[0]
#         x_fake = (x_fake + 1.0) / 2.0
#         x_real = (x_real + 1.0) / 2.0
#
#         tmp_fake = incept(x_fake)[0]
#         tmp_real = incept(x_real)[0]
#         if tmp_fake.shape[2] != 1 or tmp_fake.shape[3] != 1:
#             tmp_fake = adaptive_avg_pool2d(tmp_fake, output_size=(1, 1))
#             tmp_real = adaptive_avg_pool2d(tmp_real, output_size=(1, 1))
#
#         pred_fake[i * args.val_batch: (i + 1) * args.val_batch] = tmp_fake.cpu().data.numpy().reshape(args.val_batch, -1)
#         pred_real[i * args.val_batch: (i + 1) * args.val_batch] = tmp_real.cpu().data.numpy().reshape(args.val_batch, -1)
#
#     mu_fake = np.atleast_1d(np.mean(pred_fake, axis=0))
#     std_fake = np.atleast_2d(np.cov(pred_fake, rowvar=False))
#
#     mu_real = np.atleast_1d(np.mean(pred_real, axis=0))
#     std_real = np.atleast_2d(np.cov(pred_real, rowvar=False))
#
#     assert mu_fake.shape == mu_real.shape
#     assert std_fake.shape == std_real.shape
#
#     mu_diff = mu_fake - mu_real
#
#     covmean, _ = linalg.sqrtm(std_fake.dot(std_real), disp=False)
#
#     if not np.isfinite(covmean).all():
#         msg = ('fid calculation produces singular product; '
#                'adding %s to diagonal of cov estimates') % eps
#         print(msg)
#         offset = np.eye(std_fake.shape[0]) * eps
#         covmean = linalg.sqrtm((std_fake + offset).dot(std_real + offset))
#
#     # Numerical error might give slight imaginary component
#     if np.iscomplexobj(covmean):
#         if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
#             m = np.max(np.abs(covmean.imag))
#             raise ValueError('Imaginary component {}'.format(m))
#         covmean = covmean.real
#
#     tr_covmean = np.trace(covmean)
#
#     return mu_diff.dot(mu_diff) + np.trace(std_fake) + np.trace(std_real) - 2 * tr_covmean

