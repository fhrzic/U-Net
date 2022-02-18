import math
import random
from collections import namedtuple

import torch
from torch import nn as nn
import torch.nn.functional as F

augmentation_params = namedtuple(
    'augmentation_params',
    'flip, rotate, noise',)

class augmentation_model(nn.Module):
    def __init__(self, augmentation_params):
        super().__init__()
        self.flip = augmentation_params.flip
        self.rotate = augmentation_params.rotate
        self.noise = augmentation_params.noise
        
    def forward(self, image_batch, mask_batch):
        transfrom_t = self.transform_matrix()
        transform_t = transfrom_t.expand(image_batch.shape[0], -1, -1)
        transfrom_t = transfrom_t.to(image_batch.device, torch.float32)
        affine_t = F.affine_grid(transform_t[:, :2],
                               image_batch.size(), align_corners = False)
        
        augmented_image_batch = F.grid_sample(image_batch,
                affine_t, padding_mode='border',
                align_corners=False)
        augmented_mask_batch = F.grid_sample(mask_batch.to(torch.float32),
                affine_t, padding_mode='border',
                align_corners=False)
        import numpy as np
        #print ("TU", np.max(np.max(augmented_image_batch.detach().numpy())))
        if self.noise:
            noise_t = torch.randn_like(augmented_image_batch)
            noise_t *= self.noise

            augmented_image_batch += noise_t
        #print ("TU2", np.max(np.max(augmented_image_batch.detach().numpy())))

        augmented_image_batch = augmented_image_batch.clamp(min = 0., max = 1.)
        return augmented_image_batch, (augmented_mask_batch > 0.5).float()
        
        
    # Create affine transform matrix
    def transform_matrix(self):
        transform_t = torch.eye(3)
        
        for i in range(2):
            if self.flip:
                if random.random() > 0.5:
                    transform_t[i,i] *= -1
                    
        if self.rotate:
            angle_in_rad = random.random() * 2 * math.pi
            s = math.sin(angle_in_rad)
            c = math.cos(angle_in_rad)
            
            rotation_t = torch.tensor([
                [c, -s, 0],
                [s, c, 0],
                [0, 0 , 1]])
            
            transform_t @= rotation_t
        
        return transform_t