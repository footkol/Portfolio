import matplotlib.pyplot as plt 
import numpy as np 
import torch


def show_image(noisy_image,org_image,pred_image = None):
    
    if pred_image == None:
        
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
        
        ax1.set_title('noisy_image')
        ax1.imshow(noisy_image.permute(1,2,0).squeeze(),cmap = 'gray')
        
        ax2.set_title('original_image')
        ax2.imshow(org_image.permute(1,2,0).squeeze(),cmap = 'gray')
        
    elif pred_image != None :
        
        f, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(10,5))
        
        ax1.set_title('noisy_image')
        ax1.imshow(noisy_image.permute(1,2,0).squeeze(),cmap = 'gray')
        
        ax2.set_title('original_image')
        ax2.imshow(org_image.permute(1,2,0).squeeze(),cmap = 'gray')
        
        ax3.set_title('denoised_image')
        ax3.imshow(pred_image.permute(1,2,0).squeeze(),cmap = 'gray')
        
        
class ToTensorForAE(object):
    
    def __call__(self,sample):
        
        images,labels = sample
        
        images = images.transpose((2,0,1))
        labels = labels.transpose((2,0,1))
        
        return torch.from_numpy(images).float(),torch.from_numpy(labels).float()
