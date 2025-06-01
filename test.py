import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import models

def parse_args():
    parser = argparse.ArgumentParser(description='Test a model on CIFAR dataset')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights (.pth file)')
    parser.add_argument('--config', type=str, required=True, help='Path to model config (.yaml file)')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_transforms(transform_list):
    """Convert transform config to torchvision transforms"""
    transform_funcs = []
    for t in transform_list:
        name = t['name']
        params = t.get('params', {})
        transform_funcs.append(getattr(transforms, name)(**params))
    return transforms.Compose(transform_funcs)

def get_test_loader(config):
    # 使用配置文件中的验证集转换
    transform_test = get_transforms(config['data']['val_transforms'])
    
    test_dataset = datasets.CIFAR10(
        root=config['data']['root'],
        train=False,
        download=True,
        transform=transform_test
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    return test_loader

def test_model(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def main():
    args = parse_args()
    
    # 加载配置文件
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device(config['training']['device'])
    
    # 设置随机种子
    torch.manual_seed(config['training']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['training']['seed'])
    
    # 构建模型
    model = models.create_model(
        config['network']['name'],
        **config['network']['params']
    ).to(device)
    
    # 加载权重
    checkpoint = torch.load(args.weights, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    # 获取测试数据加载器
    test_loader = get_test_loader(config)
    
    # 测试模型
    avg_loss, accuracy = test_model(model, test_loader, device)
    
    print(f'Test Loss: {avg_loss:.4f}')
    print(f'Test Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    main() 