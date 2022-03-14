import torch
from torch import nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding = kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):
        return self.relu(self.conv(x))

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.block = [ConvLayer(in_channels, growth_rate, kernel_size=3)]
        for i in range(num_layers - 1):
            self.block.append(DenseLayer(growth_rate * (i + 1), growth_rate, kernel_size=3))
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return torch.cat([x, self.block(x)], 1)


class SR3DDenseNet(nn.Module):
    def __init__(self, num_channels=1, growth_rate=5, num_blocks=2, num_layers=4):
        super(SR3DDenseNet, self).__init__()

        # low level features
        self.conv = ConvLayer(num_channels, growth_rate * num_layers, 3)

        # high level features
        self.dense_blocks = []
        for i in range(num_blocks):
            self.dense_blocks.append(DenseBlock(growth_rate * num_layers * (i + 1), growth_rate, num_layers))
        self.dense_blocks = nn.Sequential(*self.dense_blocks)

        # bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv3d(growth_rate * num_layers + growth_rate * num_layers * num_blocks, 6, kernel_size=1),
            nn.ReLU(inplace=True)
        )
      
        # reconstruction layer
        self.reconstruction = nn.Conv3d(6, num_channels, kernel_size=3, padding=3 // 2)

        # self._initialize_weights()
       

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        # x = x.type(torch.cuda.FloatTensor)
        # print ('before first layer')
        x = self.conv(x)
        # print('first layer no error')
        x = self.dense_blocks(x)
        # print('second layer no error')
        x = self.bottleneck(x)
        # print('third layer no error')
        # x = self.deconv(x)  removed
        x = self.reconstruction(x)
        # print('fourth layer no error')
        return x