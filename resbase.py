import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True, head=True):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            #nn.LeakyReLU(0.1, inplace=True)
        )
        self.bn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # self.head = head
        self.pool = pool
        if self.pool:
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # if self.head:
        #     self.head_ly = nn.Sequential(
        #         nn.BatchNorm2d(out_channels),
        #         nn.ReLU(inplace=True)
        #     )

    def forward(self, x):
        if self.pool:
            x = self.max_pool(x)
        x = self.conv1(x)
        y = self.conv2(x)
        y = y + x
        y = self.bn(y)
        return y

class ResBase(nn.Module):

    def __init__(self):
        super(ResBase, self).__init__()
        self.layer0 = ConvBlock(3, 32, False, False)
        self.layer1 = ConvBlock(32, 64, True, True)
        self.layer2 = ConvBlock(64, 128, True, True)
        self.layer3 = ConvBlock(128, 256, True, True)
        self.layer4 = ConvBlock(256, 512, True, True)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def get_resbase(filters):
    layer0 = ConvBlock(3, filters[0], False, False)
    layer1 = ConvBlock(filters[0], filters[1], True, True)
    layer2 = ConvBlock(filters[1], filters[2], True, True)
    layer3 = ConvBlock(filters[2], filters[3], True, True)
    layer4 = ConvBlock(filters[3], filters[4], True, True)

    return layer0,layer1,layer2,layer3,layer4




















