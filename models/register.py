from .resnet import ResNet18, ResNet34, ResNet50
from .vgg import VGG_A, VGG_A_Dropout, VGG_A_Light, VGG_A_BatchNorm
from .densenet import DenseNet121, DenseNet169, DenseNet201, DenseNet

# Model factory
def create_model(model_name, **kwargs):
    models = {
        'ResNet18': ResNet18,
        'ResNet34': ResNet34,
        'ResNet50': ResNet50,
        'VGG_A': VGG_A,
        'VGG_A_Dropout': VGG_A_Dropout,
        'VGG_A_Light': VGG_A_Light,
        'VGG_A_BatchNorm': VGG_A_BatchNorm,
        'DenseNet121': DenseNet121,
        'DenseNet169': DenseNet169,
        'DenseNet201': DenseNet201,
        'DenseNet': DenseNet  # 自定义配置的DenseNet
    }
    
    if model_name not in models:
        raise ValueError(f'Model {model_name} not found. Available models: {list(models.keys())}')
        
    return models[model_name](**kwargs) 