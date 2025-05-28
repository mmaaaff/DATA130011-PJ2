import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation(name, activation_params):
    """
    Get activation function by name
    Args:
        name: Name of the activation function
        activation_params: Parameters for the activation function
    Returns:
        Activation function module
    """
    activations = {
        'ReLU': nn.ReLU,
        'LeakyReLU': nn.LeakyReLU,
        'GELU': nn.GELU,
        'SiLU': nn.SiLU,  # Also known as Swish
        'Mish': nn.Mish,
    }
    
    if name not in activations:
        raise ValueError(f'Activation function {name} not found. '
                        f'Available options: {list(activations.keys())}')
    
    if activation_params is not None:
        return activations[name](**activation_params)
    else:
        return activations[name]()

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, activation='ReLU', activation_params=None):
        super(BasicBlock, self).__init__()
        # if activation_params is None:
        #     activation_params = {'inplace': True}
            
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Create activation functions
        self.activation1 = get_activation(activation, activation_params)
        self.activation2 = get_activation(activation, activation_params)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.activation1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation2(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, activation='ReLU', activation_params=None):
        super(Bottleneck, self).__init__()
        # if activation_params is None:
        #     activation_params = {'inplace': True}
            
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        # Create activation functions
        self.activation1 = get_activation(activation, activation_params)
        self.activation2 = get_activation(activation, activation_params)
        self.activation3 = get_activation(activation, activation_params)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = self.activation1(self.bn1(self.conv1(x)))
        out = self.activation2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activation3(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=3, num_classes=10, 
                 base_channels=64, dropout_rate=0.1, zero_init_residual=False,
                 activation='ReLU', activation_params=None):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.current_channels = base_channels
        
        # if activation_params is None:
        #     activation_params = {'inplace': True}

        self.conv1 = nn.Conv2d(in_channels, self.current_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.current_channels)
        self.activation = get_activation(activation, activation_params)
        
        self.layer1 = self._make_layer(block, base_channels, num_blocks[0], stride=1,
                                     activation=activation, activation_params=activation_params)
        self.layer2 = self._make_layer(block, base_channels*2, num_blocks[1], stride=2,
                                     activation=activation, activation_params=activation_params)
        self.layer3 = self._make_layer(block, base_channels*4, num_blocks[2], stride=2,
                                     activation=activation, activation_params=activation_params)
        self.layer4 = self._make_layer(block, base_channels*8, num_blocks[3], stride=2,
                                     activation=activation, activation_params=activation_params)
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels*8*block.expansion, num_classes)

        # Initialize weights
        self._initialize_weights(zero_init_residual)

    def _make_layer(self, block, out_channels, num_blocks, stride, activation, activation_params):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.current_channels, out_channels, stride,
                              activation=activation, activation_params=activation_params))
            self.current_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:  
            # 将residual block的最后一层BN的gamma和beta初始化为零，即数学形式上的x = x + F(x)中的F(x)一开始为0，有利于训练
            # 出自Bag of Tricks for Image Classification with Convolutional Neural Networks
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.dropout(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)