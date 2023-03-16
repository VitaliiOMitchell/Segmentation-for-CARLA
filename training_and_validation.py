from Unet import Unet
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Data_preparetion import make_datasets 
import matplotlib.pyplot as plt
from tqdm import tqdm
from loss_functions_and_metrics import Tversky_loss, pixel_accuracy

    
        
def train_validate(model, epochs, train_loader, val_loader, opt, lr_scheduler, loss_func, device):
    train_losses = []
    train_average_losses = []
    val_losses = []
    val_average_losses = []
    epochs_range = np.arange(epochs)
    for epoch in range(epochs):
        training_loss = 0
        index_train = 0
        pixel_acc_train = 0
        validation_loss = 0
        index_val = 0
        pixel_acc_val = 0
        model.train()
        for i, (x, y) in enumerate(tqdm(train_loader)):
            x = x.to(device)
            y = y.to(device)
            
            opt.zero_grad()
            train_output = model(x)
            train_output = train_output
            tver_index_train, tver_loss_train = loss_func(train_output, y)
            pixel_train = pixel_accuracy(train_output, y, one_hot=True)
            index_train += tver_index_train.item()
            training_loss += tver_loss_train.item()
            pixel_acc_train += pixel_train.item()
            tver_loss_train.backward()
            opt.step()
            
            
        training_loss /= len(train_loader)
        index_train /= len(train_loader)
        pixel_acc_train /= len(train_loader)
        train_losses.append(training_loss)
        train_average_losses.append(np.mean(train_losses))
        
        model.eval()
        with torch.no_grad():
            for i, (x_val, y_val) in enumerate(tqdm(val_loader)):
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                val_output = model(x_val)
                tver_index_val, tver_val_loss = loss_func(val_output, y_val)
                pixel_val = pixel_accuracy(val_output, y_val, one_hot=True)
                index_val += tver_index_val.item()
                validation_loss += tver_val_loss.item()
                pixel_acc_val += pixel_val.item()       
                
        
        validation_loss /= len(val_loader)
        
        lr_scheduler.step(validation_loss)
        
        if len(val_losses) != 0:
            if validation_loss < np.min(val_losses):
                torch.save(model.state_dict(), 'F:/Python/Projects/Segmentation-for-Self-driving-cars/model.pth')
                print('Model saved...')
        
        index_val /= len(val_loader)
        pixel_acc_val /= len(val_loader)
        val_losses.append(validation_loss)
        val_average_losses.append(np.mean(val_losses))
        
        print(f'Epoch: {epoch}')
        tqdm().set_postfix(Train_loss = training_loss, 
                           Average_Train_loss = train_average_losses[-1],
                           Index_train = index_train,
                           Pixel_acc_train = pixel_acc_train)
        
        tqdm().set_postfix(Validation_loss = validation_loss, 
                           Average_Validation_loss = val_average_losses[-1],
                           Index_val = index_val,
                           Pixel_acc_val = pixel_acc_val)
        

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train loss')
    plt.plot(epochs_range, val_losses, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_average_losses, label='Average Train loss')
    plt.plot(epochs_range, val_average_losses, label='Average Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
        

if __name__ == '__main__':
    # Device
    device = 'cuda'
    
    # Model
    model = Unet(3, 13)
    model = model.to(device)
    
    # Hyper
    batch_size = 4
    epochs = 15
    lr = 0.1
    dice_loss = Tversky_loss()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = ReduceLROnPlateau(opt, mode='min', patience=3, factor=0.1)
    
    # Data
    x_path = 'F:/Python/Projects/Segmentation-for-Self-driving-cars/dataA/dataA/CameraRGB'
    y_path = 'F:/Python/Projects/Segmentation-for-Self-driving-cars/dataA/dataA/CameraSeg'
    train, val = make_datasets(x_path, y_path)
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=val, batch_size=batch_size, shuffle=True, num_workers=2)

    torch.backends.cudnn.benchmark = True
    print(train_validate(model, epochs, train_loader, val_loader, opt, lr_scheduler, dice_loss, device))
    

    