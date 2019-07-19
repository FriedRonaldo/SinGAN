from torch import nn
from torch.nn import functional as F
import torch


class Generator(nn.Module):
    def __init__(self, img_size_min, num_scale, scale_factor=4/3):
        super(Generator, self).__init__()
        self.img_size_min = img_size_min
        self.scale_factor = scale_factor
        self.num_scale = num_scale
        self.nf = 32
        self.current_scale = 0

        self.size_list = [int(self.img_size_min * scale_factor**i) for i in range(num_scale + 1)]
        print(self.size_list)

        self.sub_generators = nn.ModuleList()

        first_generator = nn.ModuleList()

        first_generator.append(nn.Sequential(nn.Conv2d(3, self.nf, 3, 1),
                                             nn.BatchNorm2d(self.nf),
                                             nn.LeakyReLU(2e-1)))
        for _ in range(3):
            first_generator.append(nn.Sequential(nn.Conv2d(self.nf, self.nf, 3, 1),
                                                 nn.BatchNorm2d(self.nf),
                                                 nn.LeakyReLU(2e-1)))

        first_generator.append(nn.Sequential(nn.Conv2d(self.nf, 3, 3, 1),
                                             nn.Tanh()))

        first_generator = nn.Sequential(*first_generator)

        self.sub_generators.append(first_generator)

    def forward(self, z, img=None):
        x_list = []
        x_first = self.sub_generators[0](z[0])
        x_list.append(x_first)

        if img is not None:
            x_inter = img
        else:
            x_inter = x_first

        for i in range(1, self.current_scale + 1):
            x_inter = F.interpolate(x_inter, (self.size_list[i], self.size_list[i]), mode='bilinear', align_corners=True)
            x_prev = x_inter
            x_inter = F.pad(x_inter, [5, 5, 5, 5], value=0)
            x_inter = x_inter + z[i]
            x_inter = self.sub_generators[i](x_inter) + x_prev
            x_list.append(x_inter)

        return x_list

    def progress(self):
        self.current_scale += 1

        if self.current_scale % 4 == 0:
            self.nf *= 2

        tmp_generator = nn.ModuleList()
        tmp_generator.append(nn.Sequential(nn.Conv2d(3, self.nf, 3, 1),
                                           nn.BatchNorm2d(self.nf),
                                           nn.LeakyReLU(2e-1)))

        for _ in range(3):
            tmp_generator.append(nn.Sequential(nn.Conv2d(self.nf, self.nf, 3, 1),
                                               nn.BatchNorm2d(self.nf),
                                               nn.LeakyReLU(2e-1)))

        tmp_generator.append(nn.Sequential(nn.Conv2d(self.nf, 3, 3, 1),
                                           nn.Tanh()))

        tmp_generator = nn.Sequential(*tmp_generator)

        if self.current_scale % 4 != 0:
            prev_generator = self.sub_generators[-1]

            # Initialize layers via copy
            if self.current_scale >= 1:
                tmp_generator.load_state_dict(prev_generator.state_dict())

        self.sub_generators.append(tmp_generator)
        print("GENERATOR PROGRESSION DONE")
