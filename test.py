import torch
from Unet import Unet
import matplotlib.pyplot as plt
from Data_preparetion import make_datasets


def prep_data(dataset, index):
    data = dataset.__getitem__(index)
    x = data[0]
    x = torch.unsqueeze(x, 0)
    y = data[1]
    
    return x, y
    
def test():
    model = Unet(3, 13)
    model.load_state_dict(torch.load('F:/Python/Projects/Segmentation-for-Self-driving-cars/model.pth'))
    test_dataset = make_datasets('F:/Python/Projects/Segmentation-for-Self-driving-cars/dataA/dataA/CameraRGB',
                                 'F:/Python/Projects/Segmentation-for-Self-driving-cars/dataA/dataA/CameraSeg', 
                                 test_only=True)
    image, mask = prep_data(test_dataset, 22)
    mask = torch.argmax(mask, 0)
    model.eval()
    with torch.no_grad():
        for i in range(4):
            image, mask = prep_data(test_dataset, i)
            mask = torch.argmax(mask, 0)
            pred = model(image)
            pred = torch.argmax(pred, 1)
            
            plt.figure(figsize=(14, 4))
            plt.subplot(1, 3, 1 )
            plt.imshow(image[0][0])
            plt.xlabel('Original image')
        
            plt.subplot(1, 3, 2)
            plt.imshow(mask)
            plt.xlabel('Mask')
            
            plt.subplot(1, 3, 3)
            plt.imshow(pred[0])
            plt.xlabel('Prediction')
            plt.show()
            
        
if __name__ == '__main__':
    test()
        