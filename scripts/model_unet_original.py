import torch
from torch import nn
import torch.nn.functional as F

# Model: padding is 0, if set to 1, NN will try to have same image size at the output
class UNet(nn.Module):
    def __init__(self, padding = 0):
        super(UNet, self).__init__()
        self.padding = padding
        # Number of in/out channels in up/down branch
        _channels_list =  [(3,64),(64,128), (128,256),
                           (256,512),(512,1024)]
        
        
        # Moduls on left branch
        self.downsampling_branch = nn.ModuleList()
        for _channels in _channels_list:
            self.downsampling_branch.append(Unet_conv_block(_channels[0],
                                                            _channels[1], 
                                                            self.padding))
        
        # Moduls on right branch
        self.upsampling_branch = nn.ModuleList()
        for _channels in reversed(_channels_list[1:]):
            self.upsampling_branch.append(Unet_upsampling_concat_block(_channels[1], 
                                                                       _channels[0], 
                                                                       self.padding))      
        
        self.last_conv = nn.Conv2d(_channels_list[0][1], 1, kernel_size = 1)
        self.last_activ = nn.Sigmoid()
        
    # Forward pass that puts all together
    def forward(self, x):
        _down_conv_blocks = []
        
        # Build left branch
        for i, conv_block in enumerate(self.downsampling_branch):
            x = conv_block(x)
            # If it is last branch, skip it
            if i != len(self.downsampling_branch) - 1:
                _down_conv_blocks.append(x)
                x = F.max_pool2d(x, 2)
       
        # Build right branch
        
        for i, up_block in enumerate(self.upsampling_branch):  
            x = up_block(x, _down_conv_blocks[-1-i])            
           
        # Add last layer
        x = self.last_conv(x)
        x = self.last_activ(x)
        
        return x

# Convolutional block without downsampling
class Unet_conv_block(nn.Module):
    def __init__(self, number_of_input_channels = None, number_of_filters = None, padding = 0):
        super(Unet_conv_block, self).__init__()
        assert number_of_input_channels != None
        assert number_of_filters != None
        _conv_block = []
        _conv_block.append(nn.Conv2d(number_of_input_channels, number_of_filters,
                                    kernel_size = 3, padding = padding))
        _conv_block.append(nn.ReLU())
        _conv_block.append(nn.Conv2d(number_of_filters, number_of_filters,
                                    kernel_size = 3, padding = padding))
        _conv_block.append(nn.ReLU())
        
        self.block = nn.Sequential(*_conv_block)
    
    def forward(self, block_input):
        _out = self.block(block_input)
        return _out

# Block that handels upsampling and next 2 convolution blocks
class Unet_upsampling_concat_block(nn.Module):
    def __init__(self, number_of_input_channels = None, number_of_filters = None, padding = 0):
        super(Unet_upsampling_concat_block, self).__init__()
        assert number_of_input_channels != None
        assert number_of_filters != None
        
        self.upsampling = nn.ConvTranspose2d(number_of_input_channels, 
                                             number_of_filters, 
                                             kernel_size = 2,
                                             stride = 2)
        
        self.conv_block = Unet_conv_block(number_of_input_channels, number_of_filters, padding)
        
    # Cropping arround the kernel centre
    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]    
    
    # Forward pass
    def forward(self, block_input, residual_input):
        _out_1 = self.upsampling(block_input)
        _out_2 = self.center_crop(residual_input, _out_1.shape[2:])
        _out = torch.cat([_out_1, _out_2], 1)
        return self.conv_block(_out)