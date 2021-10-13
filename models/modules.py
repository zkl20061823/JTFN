import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(3, 3), padding=(1, 1)),
            nn.Conv2d(1, 1, kernel_size=(5, 5), padding=(2, 2)),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(channel // reduction, channel, 1, bias=False)
        self.activate = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.fc1(self.avg_pool(x)))
        max_out = self.fc2(self.fc1(self.max_pool(x)))
        out = avg_out + max_out
        out = self.activate(out)
        return out

class GAU(nn.Module):
    def __init__(self, in_channels, use_gau=True, reduce_dim=False, out_channels=None):
        super(GAU, self).__init__()
        self.use_gau = use_gau
        self.reduce_dim = reduce_dim

        if self.reduce_dim:
            self.down_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            in_channels = out_channels

        if self.use_gau:

            self.sa = SpatialAttention()
            self.ca = ChannelAttention(in_channels)

            self.reset_gate = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=2, dilation=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x, y):
        if self.reduce_dim:
            x = self.down_conv(x)

        if self.use_gau:
            y = F.interpolate(y, x.shape[-2:], mode='bilinear', align_corners=True)

            comx = x * y
            resx = x * (1 - y) # bs, c, h, w

            x_sa = self.sa(resx) # bs, 1, h, w
            x_ca = self.ca(resx) # bs, c, 1, 1

            O = self.reset_gate(comx)
            M = x_sa * x_ca

            RF = M * x + (1 - M) * O
        else:
            RF = x
        return RF

class FIM(nn.Module):

    def __init__(self, in_channels, out_channels, f_channels, use_topo=True, up=True, bottom=False):
        super(FIM, self).__init__()
        self.use_topo = use_topo
        self.up = up
        self.bottom = bottom

        if self.up:
            self.up_s = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.up_t = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.up_s = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.up_t = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.decoder_s = nn.Sequential(
            nn.Conv2d(out_channels + f_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.inner_s = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        if self.bottom:
            self.st = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )

        if self.use_topo:
            self.decoder_t = nn.Sequential(
                nn.Conv2d(out_channels + out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

            self.s_to_t = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

            self.t_to_s = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

            self.res_s = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

            self.inner_t = nn.Sequential(
                nn.Conv2d(out_channels, 1, kernel_size=3, padding=1, bias=False),
                nn.Sigmoid()
            )

    def forward(self, x_s, x_t, rf):
        if self.use_topo:
            if self.bottom:
                x_t = self.st(x_t)
            #bs, c, h, w = x_s.shape
            x_s = self.up_s(x_s)
            x_t = self.up_t(x_t)

            # padding
            diffY = rf.size()[2] - x_s.size()[2]
            diffX = rf.size()[3] - x_s.size()[3]

            x_s = F.pad(x_s, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
            x_t = F.pad(x_t, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])

            rf_s = torch.cat((x_s, rf), dim=1)
            s = self.decoder_s(rf_s)
            s_t = self.s_to_t(s)

            t = torch.cat((x_t, s_t), dim=1)
            x_t = self.decoder_t(t)
            t_s = self.t_to_s(x_t)

            s_res = self.res_s(torch.cat((s, t_s), dim=1))

            x_s = s + s_res
            t_cls = self.inner_t(x_t)
            s_cls = self.inner_s(x_s)
        else:
            x_s = self.up_s(x_s)
            #x_b = self.up_b(x_b)
            # padding
            diffY = rf.size()[2] - x_s.size()[2]
            diffX = rf.size()[3] - x_s.size()[3]

            x_s = F.pad(x_s, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])

            rf_s = torch.cat((x_s, rf), dim=1)
            s = self.decoder_s(rf_s)
            x_s = s
            x_t = x_s
            t_cls = None
            s_cls = self.inner_s(x_s)
        return x_s, x_t, s_cls, t_cls

class BaseDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, f_channels):
        super(BaseDecoder, self).__init__()

        self.up_conv = nn.ConvTranspose2d(in_channels, f_channels, kernel_size=2, stride=2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(f_channels * 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, rf):

        x = self.up_conv(x, output_size=rf.size())

        #padding
        diffY = rf.size()[2] - x.size()[2]
        diffX = rf.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        y = self.conv1(torch.cat([x, rf], dim=1))
        y = self.conv2(y)

        return y












