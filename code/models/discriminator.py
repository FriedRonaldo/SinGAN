from torch import nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.nf = 32
        self.current_scale = 0

        self.sub_discriminators = nn.ModuleList()

        first_discriminator = nn.ModuleList()

        first_discriminator.append(nn.Sequential(nn.Conv2d(3, self.nf, 3, 1, 1),
                                             nn.LeakyReLU(2e-1)))
        for _ in range(3):
            first_discriminator.append(nn.Sequential(nn.Conv2d(self.nf, self.nf, 3, 1, 1),
                                                 nn.BatchNorm2d(self.nf),
                                                 nn.LeakyReLU(2e-1)))

        first_discriminator.append(nn.Sequential(nn.Conv2d(self.nf, 1, 3, 1, 1)))

        first_discriminator = nn.Sequential(*first_discriminator)

        self.sub_discriminators.append(first_discriminator)

    def forward(self, x):
        out = self.sub_discriminators[self.current_scale](x)
        return out

    def progress(self):
        self.current_scale += 1
        # Lower scale discriminators are not used in later ... replace append to assign?
        if self.current_scale % 4 == 0:
            self.nf *= 2

        tmp_discriminator = nn.ModuleList()
        tmp_discriminator.append(nn.Sequential(nn.Conv2d(3, self.nf, 3, 1, 1),
                                               nn.LeakyReLU(2e-1)))

        for _ in range(3):
            tmp_discriminator.append(nn.Sequential(nn.Conv2d(self.nf, self.nf, 3, 1, 1),
                                                   nn.BatchNorm2d(self.nf),
                                                   nn.LeakyReLU(2e-1)))

        tmp_discriminator.append(nn.Sequential(nn.Conv2d(self.nf, 1, 3, 1, 1)))

        tmp_discriminator = nn.Sequential(*tmp_discriminator)

        if self.current_scale % 4 != 0:
            prev_discriminator = self.sub_discriminators[-1]

            # Initialize layers via copy
            if self.current_scale >= 1:
                tmp_discriminator.load_state_dict(prev_discriminator.state_dict())

        self.sub_discriminators.append(tmp_discriminator)
        print("DISCRIMINATOR PROGRESSION DONE")
