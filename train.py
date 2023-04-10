import sys
import pandas
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import time
import models
import argparse
import numpy as np
from dataset import train_dataloader, val_dataloader
from dataset import train_dataset, val_dataset
from utils import EarlyStopping, LRScheduler
from tqdm import tqdm

matplotlib.style.use('ggplot')

# Empty cache
torch.cuda.empty_cache()

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--lr-scheduler', dest='lr_scheduler', action='store_true')
parser.add_argument('--early-stopping', dest='early_stopping', action='store_true')
args = vars(parser.parse_args())

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")
# instantiate the model 
model = models.Resnet18_UNet(n_channels=3).to(device)
model = models.UNetWithResnet50Encoder().to(device)
# total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

# number epochs
epochs = 20
# optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-6)
# loss function
criterion = nn.MSELoss()

# strings to save the loss plot, accuracy plot, and model with different ...
# ... names according to the training type
# if not using `--lr-scheduler` or `--early-stopping`, then use simple names
loss_plot_name = 'loss'
model_name = 'Unet'

# either initialize early stopping or learning rate scheduler
if args['lr_scheduler']:
    print('INFO: Initializing learning rate scheduler')
    lr_scheduler = LRScheduler(optimizer)
    # change the accuracy, loss plot names and model name
    loss_plot_name = 'lrs_loss'
    acc_plot_name = 'lrs_accuracy'
    model_name = 'lrs_model'
if args['early_stopping']:
    print('INFO: Initializing early stopping')
    early_stopping = EarlyStopping()
    # change the accuracy, loss plot names and model name
    loss_plot_name = 'es_loss'
    acc_plot_name = 'es_accuracy'
    model_name = 'es_model'


def fit(model, train_dataloader, train_dataset, optimizer, criterion):
    print('Training')
    model.train()
    train_running_loss = 0.0
    counter = 0
    prog_bar = tqdm(enumerate(train_dataloader), total=int(len(train_dataset) / train_dataloader.batch_size))
    for i, data in prog_bar:
        counter += 1
        data = data[0].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, data)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / counter
    return train_loss


def validate(model, val_dataloader, val_dataset, criterion):
    print('\nValidating')
    model.eval()
    val_running_loss = 0.0
    counter = 0
    prog_bar = tqdm(enumerate(val_dataloader), total=int(len(val_dataset) / val_dataloader.batch_size))
    with torch.no_grad():
        for i, data in prog_bar:
            counter += 1
            data, labels = data[0].to(device), data[1].to(device)
            outputs = model(data)
            loss = criterion(outputs, data)
            val_running_loss += loss.item()

        val_loss = val_running_loss / counter

        return val_loss


def print_time_elapsed(start_time):
    end = time.time()
    hours, rem = divmod(end - start_time, 3600)
    minutes, seconds = divmod(rem, 60)

    if hours > 0:
        print(f"Training time: {hours}hrs {minutes}min {seconds:.0f}sec")
    else:
        print(f"Training time: {minutes}min {seconds:.0f}sec")


# lists to store per-epoch loss values
train_loss = []
val_loss = []
start = time.time()
best_val_loss = sys.float_info.max
for epoch in range(epochs):
    print(f"Epoch {epoch + 1} of {epochs}")
    train_epoch_loss = fit(model, train_dataloader, train_dataset, optimizer, criterion)
    val_epoch_loss = validate(model, val_dataloader, val_dataset, criterion)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)

    if args['lr_scheduler']:
        lr_scheduler(val_epoch_loss)
    if args['early_stopping']:
        early_stopping(val_epoch_loss)
        if early_stopping.early_stop:
            break

    print(f"Train Loss: {train_epoch_loss:.6f}")
    print(f"Val Loss: {val_epoch_loss:.6f}")

    # Save the best model according to val_score rule below
    if best_val_loss > val_epoch_loss:
        best_val_loss = val_epoch_loss
        print(f"Saving best model until now with validation_loss = {best_val_loss:.6f}")
        torch.save(model.state_dict(), f"C:\\Users\\AntonioM\\PycharmProjects\\autoencoder2\\outputs\\{model_name}.pth")

print_time_elapsed(start)
print('Saving metrics and loss plot...')

# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='black', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"C:\\Users\\AntonioM\\PycharmProjects\\autoencoder2\\outputs\\{loss_plot_name}.png")
plt.show()

# Save metrics into a file
df = pandas.DataFrame({'train_loss': train_loss, 'val_loss': val_loss})
df.index = np.arange(1, len(df)+1)  # so index starts at one
df.to_excel(f"C:\\Users\\AntonioM\\PycharmProjects\\autoencoder2\\outputs\\metrics.xlsx")

print('TRAINING COMPLETE')
