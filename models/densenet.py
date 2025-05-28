import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, activation='ReLU', activation_params=None):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.activation1 = get_activation(activation, activation_params)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate,
                              kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.activation2 = get_activation(activation, activation_params)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = float(drop_rate)

    def bn_function(self, inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.activation1(self.norm1(concated_features)))
        return bottleneck_output

    def forward(self, x):
        # note: Input x should be a list of tensors
        if isinstance(x, torch.Tensor):
            prev_features = [x]
        else:
            prev_features = x

        bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.activation2(self.norm2(bottleneck_output)))
        
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        
        return new_features

class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, 
                 drop_rate, activation='ReLU', activation_params=None):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                activation=activation,
                activation_params=activation_params
            )
            # note: 添加到ModuleDict中
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        # note: 遍历ModuleDict中的每一层
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)  # 所有层的输出拼接在一起作为block的输出

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, activation='ReLU', activation_params=None):
        super(_Transition, self).__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.activation = get_activation(activation, activation_params)
        self.conv = nn.Conv2d(num_input_features, num_output_features,
                             kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

def get_activation(name, activation_params):
    """Get activation function by name"""
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

class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0,
                 num_classes=10, in_channels=3, activation='ReLU',
                 activation_params=None, compression=2):
        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=3, 
                               stride=1, padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', get_activation(activation, activation_params)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                activation=activation,
                activation_params=activation_params
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:  # 如果不是最后一个denseblock，则添加transition层
                out_features = num_features // compression
                trans = _Transition(num_input_features=num_features,
                                  num_output_features=out_features,
                                  activation=activation,
                                  activation_params=activation_params)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = out_features

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('activation5', get_activation(activation, activation_params))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                      nonlinearity='relu' if activation == 'ReLU' else 'linear')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def DenseNet121(**kwargs):
    return DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)

def DenseNet169(**kwargs):
    return DenseNet(growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)

def DenseNet201(**kwargs):
    return DenseNet(growth_rate=32, block_config=(6, 12, 48, 32), **kwargs)