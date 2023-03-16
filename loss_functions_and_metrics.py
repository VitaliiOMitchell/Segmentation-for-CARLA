import torch
#import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Dice_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def forward(self, preds, target):
        preds = F.softmax(preds)
        batch = preds.shape[0]
        smooth = 1
        
        preds = preds.reshape(batch, -1)
        target = target.reshape(batch, -1)
        intersection = (preds*target).sum()
        union = preds.sum() + target.sum()
        
        dice_score = ((2 * intersection + smooth) / (union + smooth)) / batch
        dice_loss = 1 - dice_score
        
        return dice_score, dice_loss
    

class Tversky_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def forward(self, preds, target, alpha=0.3, beta=0.7):
        preds = F.softmax(preds)
        batch = preds.shape[0]
        smooth = 1
        
        preds = preds.reshape(batch, -1)
        target = target.reshape(batch, -1)
        TP = (preds*target).sum()
        FN = ((1-preds) * target).sum()
        FP = ((1-target) * preds).sum()
        
        tversky_index = TP / (TP + alpha*FN + beta*FP + smooth)
        tversky_loss = 1 - tversky_index
        
        return tversky_index, tversky_loss
        
        
def pixel_accuracy(pred, target, one_hot=False):
    pred = F.softmax(pred)
    if one_hot:
        preds = torch.where(pred>=0.5, 1, 0)
    else:
        preds = torch.argmax(pred, 1)
    preds = preds.reshape(-1)
    target = target.reshape(-1)
    correct = (preds==target).sum()
    return correct / len(target)
    
    
'''if __name__ == '__main__':
    pred = torch.randn(2, 4, 100, 100)
    print(F.softmax(pred, dim=1))'''
    
    
    
    
    
    
    
    
    
        
        