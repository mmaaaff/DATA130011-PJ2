import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import sys
import logging

import models

class TeeLogger:
    """同时将输出写入到控制台和文件的日志工具"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        sys.stdout = self  # 将 Python 的标准输出重定向到 TeeLogger 对象本身。这样，任何通过 print 输出的内容都会调用 TeeLogger 的 write 方法，而不是直接输出到终端。

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        sys.stdout = self.terminal  # 将标准输出恢复为原始终端
        self.log.close()

def setup_logging(config):
    """Setup experiment logging"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')   # 时间戳字符串，YYYYMMDD_HHMMSS
    save_dir = Path(config['logging']['save_dir']) / timestamp  # Path()转换为Path对象，/进行连接
    save_dir.mkdir(parents=True, exist_ok=True)  # 自动创建父目录及覆盖
    
    # Save config
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Setup console and file logging
    log_file = save_dir / 'training.log'
    logger = TeeLogger(log_file)
    
    # Print initial information
    print(f"Starting training at {timestamp}")
    print(f"Saving outputs to {save_dir}")
    print("\nConfiguration:")
    print(yaml.dump(config))
    
    # Setup tensorboard
    writer = None
    if config['logging']['tensorboard']:
        writer = SummaryWriter(str(save_dir / 'tensorboard'))
        
    return save_dir, writer, logger

def parse_args():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_transforms(transform_list):
    """Convert transform config to torchvision transforms"""
    transform_funcs = []
    for t in transform_list:
        name = t['name']  # 转换名称，如'RandomCrop'
        params = t.get('params', {})  # 转换参数，如size=32
        transform_funcs.append(getattr(transforms, name)(**params))  # 使用getattr动态获取transforms中的函数
    return transforms.Compose(transform_funcs)

def get_optimizer(model, config):
    """Create optimizer from config"""
    optimizer_config = config['optimizer']
    optimizer_class = getattr(optim, optimizer_config['type'])
    optimizer = optimizer_class(model.parameters(), **optimizer_config['params'])
    
    # Create scheduler if specified
    scheduler = None
    if 'scheduler' in optimizer_config:
        scheduler_config = optimizer_config['scheduler']
        scheduler_class = getattr(optim.lr_scheduler, scheduler_config['type'])
        scheduler = scheduler_class(optimizer, **scheduler_config['params'])
    
    return optimizer, scheduler


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, config):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()  # note
        
        if batch_idx % config['logging']['log_freq'] == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
        if writer:
            writer.add_scalar('train/batch_loss', loss.item(), 
                            epoch * len(train_loader) + batch_idx)
                
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    if writer:
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        writer.add_scalar('train/epoch_accuracy', accuracy, epoch)
        
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device, epoch, writer):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    if writer:
        writer.add_scalar('val/epoch_loss', avg_loss, epoch)
        writer.add_scalar('val/epoch_accuracy', accuracy, epoch)
        
    return avg_loss, accuracy

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Set random seed
    torch.manual_seed(config['training']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['training']['seed'])
    
    # Setup device
    device = torch.device(config['training']['device'])
    
    # Setup logging
    save_dir, writer, logger = setup_logging(config)
    
    try:
        # Setup data loaders
        train_transform = get_transforms(config['data']['train_transforms'])
        val_transform = get_transforms(config['data']['val_transforms'])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root=config['data']['root'], train=True, download=True, transform=train_transform)
        val_dataset = torchvision.datasets.CIFAR10(
            root=config['data']['root'], train=False, download=True, transform=val_transform)
        
        train_loader = DataLoader(
            train_dataset, batch_size=config['training']['batch_size'],
            shuffle=True, num_workers=config['training']['num_workers'])
        val_loader = DataLoader(
            val_dataset, batch_size=config['training']['batch_size'],
            shuffle=False, num_workers=config['training']['num_workers'])
        
        # Create model
        model = models.create_model(
            config['network']['name'],
            **config['network']['params']
        ).to(device)
        print("\nModel architecture:")
        # print(model)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer, scheduler = get_optimizer(model, config)
        
        # Training loop
        best_acc = 0
        for epoch in range(config['training']['epochs']):
            print(f'\nEpoch: {epoch}')
            
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer,
                device, epoch, writer, config)
            
            val_loss, val_acc = validate(
                model, val_loader, criterion,
                device, epoch, writer)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save checkpoint
            if epoch % config['logging']['save_freq'] == 0:
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'config': config
                }
                checkpoint_path = save_dir / f'checkpoint_epoch_{epoch}.pth'
                torch.save(state, checkpoint_path)
                print(f'Saved checkpoint to {checkpoint_path}')
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_path = save_dir / 'best_model.pth'
                torch.save(state, best_model_path)
                print(f'New best accuracy: {best_acc:.2f}%')
                print(f'Saved best model to {best_model_path}')
            
            if scheduler is not None:
                scheduler.step()
                print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        print(f'\nTraining completed. Best validation accuracy: {best_acc:.2f}%')
        
    except Exception as e:
        print(f'\nError occurred during training: {str(e)}')
        raise
    
    finally:
        if writer:
            writer.close()
        logger.close()

if __name__ == '__main__':
    main() 