import torch.nn as nn

class TransitionLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    def __init__(self, in_channels, num_classes, block_config):
        super(DenseNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dense_block_layers = nn.Sequential()

        block_in_channels = in_channels
        growth_rate = 32
        for i, layers_counts in enumerate(block_config):
            block = DenseBlock(in_channels=block_in_channels, layer_counts=layers_counts, growth_rate=growth_rate)
            self.dense_block_layers.add_module('block%d' % (i + 1), block)

            block_out_channels = block_in_channels + layers_counts * growth_rate
            transition = TransitionLayer(block_out_channels, block_out_channels // 2)
            if i != len(block_config):  # 最后一个dense block后没有transition layer
                self.dense_block_layers.add_module('transition%d' % (i + 1), transition)

            block_in_channels = block_out_channels // 2  # 更新下一个dense block的in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(block_in_channels, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(x)
        for layer in self.dense_block_layers:
            out = layer(out)
            # print(out.shape)
        out = self.avg_pool(out)
        out = torch.flatten(out, start_dim=1)  # 相当于out = out.view((x.shape[0],-1))
        out = self.fc(out)
        return out

