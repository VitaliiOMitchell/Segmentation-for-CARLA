import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as opt
import torchvision
from sys import getsizeof
from torchvision import transforms
import torchvision.transforms as TF

def crop_tensor(tensor, target_tensor):
    tensor_size = tensor.shape[2]
    target_size = target_tensor.shape[2]
    delta = (tensor_size - target_size) // 2
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]

class Down(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = nn.Conv2d(out_chan, out_chan, 3, 1, 1)
        #Batch norm

    def forward(self, input):
        output = self.conv1(input)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = F.relu(self.bn1(output))
        
        return output
    
class Up(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, 3, 1, 1)
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.1)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = nn.Conv2d(out_chan, out_chan, 3, 1, 1)
        # Batch norm

    def forward(self, input):
        output = self.conv1(input)
        output = F.relu(self.bn1(output))
        #output = self.drop1(output)
        output = self.conv2(output)
        output = F.relu(self.bn1(output))
        #output = self.drop1(output)
        #output = self.conv2(output)
        #output = F.relu(self.bn1(output))
        #output = self.drop2(output)
        #output = self.conv2(output)
        #output = F.relu(self.bn1(output))
        #output = self.drop2(output)
        
        return output
    
class Bottleneck(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, 3, 1, 1)
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.1)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = nn.Conv2d(out_chan, out_chan, 3, 1, 1)
        # Batch norm

    def forward(self, input):
        output = self.conv1(input)
        output = F.relu(self.bn1(output))
        #output = self.drop1(output)
        output = self.conv2(output)
        output = F.relu(self.bn1(output))
        #output = self.drop1(output)
        #output = self.conv2(output)
        #output = F.relu(self.bn1(output))
        #output = self.drop2(output)
        #output = self.conv2(output)
        #output = F.relu(self.bn1(output))
        #output = self.drop2(output)

        return output


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__() 
        # Down
        self.down1 = Down(in_channels, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.pool = nn.MaxPool2d(2, 2, 0)

        # Bottleneck
        self.bottle = Bottleneck(512, 1024)
        
        # Up
        self.trans_layer1 = nn.ConvTranspose2d(1024, 512, 2, 2, 0, 0)
        self.up1 = Up(1024, 512)
        self.trans_layer2 = nn.ConvTranspose2d(512, 256, 2, 2, 0, 0)
        self.up2 = Up(512, 256)
        self.trans_layer3 = nn.ConvTranspose2d(256, 128, 2, 2, 0, 0)
        self.up3 = Up(256, 128)
        self.trans_layer4 = nn.ConvTranspose2d(128, 64, 2, 2, 0, 0)
        self.up4 = Up(128, 64)
        self.last_conv = nn.Conv2d(64, out_channels, 1, 1, 0)


    def forward(self, input):
        # Down 1
        connection1 = self.down1(input)
        down1 = self.pool(connection1)
        # Down 2
        connection2 = self.down2(down1)
        down2 = self.pool(connection2)
        # Down 3
        connection3 = self.down3(down2)
        down3 = self.pool(connection3)
        # Down 4 
        connection4 = self.down4(down3)
        down4 = self.pool(connection4)
        
        # Bottleneck
        bottle = self.bottle(down4)

        # Up 1
        up1 = self.trans_layer1(bottle)
        crop1 = crop_tensor(connection4, up1)
        up1 = torch.cat([up1, crop1], 1)
        up1 = self.up1(up1)
        # Up 2
        up2 = self.trans_layer2(up1)
        crop2 = crop_tensor(connection3, up2)
        up2 = torch.cat([up2, crop2], 1)
        up2 = self.up2(up2)
        # Up 3
        up3 = self.trans_layer3(up2)
        crop3 = crop_tensor(connection2, up3)
        up3 = torch.cat([up3, crop3], 1)
        up3 = self.up3(up3)
        # Up 4
        up4 = self.trans_layer4(up3)
        crop4 = crop_tensor(connection1, up4)
        up4 = torch.cat([up4, crop4], 1)
        up4 = self.up4(up4)
        # Output 
        output = self.last_conv(up4)
        
        return output
    
#if __name__ == '__main__':
    #unet = Unet(3, 13)
    #arr = torch.randn(1,3,256,256)
    #res = unet(arr)
    #print(res.shape)
    #print(summary(unet, arr))