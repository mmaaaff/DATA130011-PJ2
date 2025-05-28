import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm # you need to implement this network
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
device_id = [0,1,2,3]
num_workers = 4
batch_size = 128

# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

# Make sure you are using the right device.
device_id = device_id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device("cuda:{}".format(3) if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(3))



# Initialize your data loader and
# make sure that dataloader works
# as expected by observing one
# sample from it.
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)
for X,y in train_loader:
    print("Shape of X of one batch", X.shape)  # [b, c, h, w]
    print("Shape of y of one batch", y.shape)  # [b]
    print("First 10 labels", y[:10])  # first 10 labels
    break



# This function is used to calculate the accuracy of model classification
def get_accuracy(model, data_loader):
    """
    Calculate the accuracy of the model on a given dataset
    Args:
        model: the neural network model
        data_loader: data loader for either training or validation dataset
    Returns:
        accuracy: the accuracy of the model on the dataset
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def get_loss(model, data_loader, criterion):
    
    model.eval()
    loss = 0
    
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss += criterion(outputs, labels)
    
    loss /= len(data_loader)
    return loss


# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader:torch.utils.data.DataLoader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    batch_size = train_loader.batch_size
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    val_loss_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    loss_list = []  # use this to record the loss value of each step
    grad = []  # use this to record the loss gradient of each step
    
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()
        learning_curve[epoch] = 0  # maintain this to plot the training curve

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            # You may need to record some variable values here
            # if you want to get loss gradient, use
            # grad = model.classifier[4].weight.grad.clone()

            # Record loss value
            loss_list.append(loss.item())
            learning_curve[epoch] += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Record gradients
            if hasattr(model, 'classifier') and len(model.classifier) > 4:
                current_grad = model.classifier[4].weight.grad.clone()
                grad.append(current_grad.cpu().numpy())
            
            optimizer.step()

        
        # Calculate training and validation accuracy
        learning_curve[epoch] /= batches_n
        train_accuracy = get_accuracy(model, train_loader)
        val_accuracy = get_accuracy(model, val_loader)
        val_loss = get_loss(model, val_loader, criterion)
        
        train_accuracy_curve[epoch] = train_accuracy
        val_accuracy_curve[epoch] = val_accuracy
        val_loss_curve[epoch] = val_loss
        
        # Save best model
        if val_accuracy > max_val_accuracy and best_model_path is not None:
            max_val_accuracy = val_accuracy
            max_val_accuracy_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
        
        # Display training progress
        display.clear_output(wait=True)
        f, axes = plt.subplots(1, 2, figsize=(15, 3))
    
    
    x = np.linspace(batch_size, batch_size * batches_n - 1, batches_n)
    for e in range(1, epochs_n):
        x_ = np.linspace(batch_size, batch_size * batches_n - 1, batches_n) + e * len(data)
        x = np.concatenate(x, x_)


    axes[0].plot(x, loss_list, label='Training Loss')
    axes[0].plot(val_loss_curve, label='Training Loss')
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    axes[1].plot(train_accuracy_curve, label='Train Accuracy')
    axes[1].plot(val_accuracy_curve, label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'training_progress_epoch_{epoch}.pdf', format='pdf', bbox_inches='tight')
    plt.show()
        
    return loss_list, grads


# Train your model
# feel free to modify
epo = 20
loss_save_path = ''
grad_save_path = ''

set_random_seeds(seed_value=2020, device=device)
lrs = [1e-3, 2e-3, 1e-4, 5e-4]
criterion = nn.CrossEntropyLoss()
loss_record = []

for lr in lrs:
    model = VGG_A()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    loss, grads = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
    loss_record.append(loss)
    np.savetxt(os.path.join(loss_save_path, f'loss_{lr}.txt'), loss, fmt='%s', delimiter=' ')
    np.savetxt(os.path.join(grad_save_path, f'grads.txt_{lr}'), grads, fmt='%s', delimiter=' ')

# Maintain two lists: max_curve and min_curve,
# select the maximum value of loss in all models
# on the same step, add it to max_curve, and
# the minimum value to min_curve
min_curve = []
max_curve = []
## --------------------
# Add your code
#
#
#
#
## --------------------

# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()
def plot_loss_landscape(min_curve, max_curve):
    """
    Plot the loss landscape using the min and max curves
    Args:
        min_curve: list of minimum loss values at each step
        max_curve: list of maximum loss values at each step
    """
    plt.figure(figsize=(10, 6))
    steps = range(len(min_curve))
    
    plt.plot(steps, max_curve, 'r--', label='Maximum Loss')
    plt.plot(steps, min_curve, 'b--', label='Minimum Loss')
    plt.fill_between(steps, min_curve, max_curve, alpha=0.2, color='gray')
    
    plt.xlabel('Steps')
    plt.ylabel('Loss Value')
    plt.title('Loss Landscape')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_landscape.png')
    plt.close()