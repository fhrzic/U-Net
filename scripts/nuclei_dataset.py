import torch
import torch.cuda
from torch.utils.data import Dataset
import glob
import os
import functools
import sys
from utils.logging import logging
import SimpleITK as sitk
from collections import namedtuple
from PIL import Image
from scipy import ndimage
import copy
import numpy as np
from utils.disk import getCache
from skimage.transform import resize
import random

# Cashing script
raw_cache = getCache('NucleiCache')

# Logging
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)     
        
        
nuclei_info_tuple = namedtuple(
    'nuclei_info_tuple',
    'studyId, maskId, cx, cy',
)



@functools.lru_cache(2)
def get_nuclei_info(nuclei_dir_path = None, nuclei_mask_combined = True):
    """
        Getting list of nuclei on the hard disk. Every sample has: path to image, path to masks, center of nuclei if nuclei_mask_combined = False. 
        If nuclei_mask_combined = False, for each nuceli in every image one input is made.

        Args:
            * nuclei_dir_path: string, path to data

            * nuclei_mask_combined, boolean, True if all masks are merged, False for per-nuclei patch mode
    """
    # Checking if nuclei_dir_path is defined, and if it is-extracting data names to list
    try: 
        if nuclei_dir_path != None:
            _nuclei_id_list = [os.path.basename(x) for x in glob.glob(nuclei_dir_path)]
        else:
            raise ValueError    
    except ValueError: 
        print("Nuclei dir path is None, data will not be loaded!")   
        sys.exit(1) 
    
    # Logging
    log.info("Searching for samples..., this can take a while.")  
    
    # Getting masks to save all data in format study:[(mask, center, radious)]
    _nuclei_list = []
    for _nuclei_id in _nuclei_id_list:
        # Case there is only one mask, the mask will be generated automatically
        if nuclei_mask_combined:
            _nuclei_info = nuclei_info_tuple(f"{nuclei_dir_path[:-1]}{_nuclei_id}/images/{_nuclei_id}.png", 
                                             '/', '/', '/')
            _nuclei_list.append(_nuclei_info)
        else:
            # Separated masks
            # Get masks
            _mask_file_list = [os.path.join(f"{nuclei_dir_path[:-1]}{_nuclei_id}/masks/", file) 
                          for file in os.listdir(f"{nuclei_dir_path[:-1]}{_nuclei_id}/masks/")]
            
            for _i , _mask_file in enumerate(_mask_file_list):
                # Load data
                _np_img = np.asarray(Image.open(_mask_file))

                # Get cetroid and estimated radious
                _cy, _cx = ndimage.center_of_mass(_np_img)

                # Save
                _nuclei_info = nuclei_info_tuple(f"{nuclei_dir_path[:-1]}{_nuclei_id}/images/{_nuclei_id}.png", 
                                                   _mask_file, _cx, _cy)

                _nuclei_list.append(_nuclei_info)
            
    return _nuclei_list

@functools.lru_cache(maxsize=1, typed=True)
def get_nuclei(instance, nuclei_combined = False, patch_size = 64):
    """
        Help function for cashing.
    """
    return Nuclei(instance, nuclei_combined, patch_size)

@raw_cache.memoize(typed=True)
def get_nuclei_sample(instance, nuclei_combined = False, patch_size = 64):
    """
        Help function for cashing.
    """
    _nuclei = get_nuclei(instance, nuclei_combined, patch_size)
    _img, _mask = _nuclei.get_sample()
    return _img, _mask


class Nuclei:
    """
        Class for handeling nuclei
    """
    def __init__(self, instance, nuclei_combined = False, patch_size = 64):
        """
            Creating nuclei object and everything about it: input data, mask, center of nuclei when nuclei_combined is set to False

            Args:
                * Instance: string, path to nuclei
                
                * nuclei_combined: boolean, how to generate mask

                * patch_size, int, how big the input image is (nuclei_combined = True), or crop arround 
                nuclei center (nuclei_combined = False)
                
        """
        
        self.nuclei_combined = nuclei_combined
        self.instance = instance
        self.patch_size = patch_size
        
        # Genrate image
        self.input_image = sitk.ReadImage(instance.studyId)
        self.input_image = np.array(sitk.GetArrayFromImage(self.input_image), dtype=np.float32)

        # Generate mask
        _mask_folder = '/'.join(self.instance.studyId.split('/')[:-2])+'/masks/'       
        _mask_file_list = [os.path.join(_mask_folder, _file) 
                      for _file in os.listdir(_mask_folder)]
        # Create output mask
        self.output_mask = np.zeros(self.input_image.shape[:2])
        for _mask_file in _mask_file_list:
            _mask = sitk.ReadImage(_mask_file)
            _mask = np.array(sitk.GetArrayFromImage(_mask), dtype=np.float32)
            self.output_mask += _mask
    
    # Full image mode
    def get_sample_full_image(self):
        """
            Creating images to desired size for nuclei_combined = true
        """
        _img = self.input_image
        _mask = self.output_mask
        
        # Padding
        _height = self.input_image.shape[0]
        _width = self.input_image.shape[1]
        
        _pad_width = int(abs((_height-_width)) / 2)
        if _pad_width % 2 == 0:
            _before = _pad_width
            _after = _pad_width
        else:
            _before = _pad_width
            _after = _pad_width + 1

        if _height > _width:
            _img = np.pad(_img, pad_width=[(0, 0),(_before, _after),(0, 0)], mode='constant')
            _mask = np.pad(_mask, pad_width=[(0, 0),(_before, _after)], mode='constant')

        if _width > _height:
            _img = np.pad(_img, pad_width=[(_before, _after),(0, 0), (0, 0)], mode='constant')
            _mask = np.pad(_mask, pad_width=[(_before, _after), (0, 0)], mode='constant')
        
        # Resizing
        _img = resize(_img, (self.patch_size, self.patch_size))
        _mask = resize(_mask, (self.patch_size, self.patch_size))

        return _img, _mask
    
    # Only patches
    def get_sample_patch_image(self):
        """
            Creating patches for mode nuclei_combined = false
        """

        # Check for correct patch size
        try: 
            if self.patch_size % 2 == 0 and self.nuclei_combined == False:
                _patch_size = int(self.patch_size / 2)
            else:
                raise ValueError    
        except ValueError: 
            print("Patch size must be even number!")   
            sys.exit(1) 
        
        # Crop mask and image
        _height = self.input_image.shape[0]
        _width = self.input_image.shape[1]
        _x = int(self.instance.cx)
        _y = int(self.instance.cy)

        if _x - _patch_size < 0:
            _x = _patch_size
        if _x + _patch_size > _width:
            _x = _width-_patch_size
        if _y - _patch_size < 0:
            _y = _patch_size
        if _y + _patch_size > _height:
            _y = _height-_patch_size
        
        _mask = self.output_mask[_y-_patch_size:_y+_patch_size, _x-_patch_size:_x+_patch_size]
        _img = self.input_image[_y-_patch_size:_y+_patch_size, _x-_patch_size:_x+_patch_size]
        return _img, _mask
    
    
    def get_sample(self):
        """
            Handler function for grabbing samples
        """
        if self.nuclei_combined:
            _image, _mask = self.get_sample_full_image()
        else:
            _image, _mask = self.get_sample_patch_image()
        return _image, _mask



# Main class for dataset 
class nuclei_dataset:

    def __init__(self, nuclei_dir_path = None, instance = None, nuclei_combined = False, patch_size = 64):
        """ Init: path to Nuclei data and mode, instance (path to instance or tupple(instance,mask)),
            combined mask or separate)

            Args:

                * nuclei_dir_path, str, path to data
                
                * instance, str, path to one instance-dataset is just that instance, 
                
                * nuclei_combined, boolean, masks combined or per nucleia
                
                * patch_size, int , size of image / patch
        """
        try: 
            if nuclei_dir_path != None:
                self.nuclei_info_data = copy.copy(get_nuclei_info(nuclei_dir_path, nuclei_combined))
            else:
                raise ValueError    
        except ValueError: 
            print("Nuclei dir path is None, data will not be loaded!")   
            sys.exit(1) 
              
        self.nuclei_combined = nuclei_combined
        self.patch_size = patch_size
        # Reduce list if only one insance is needed
        if instance:
            if self.nuclei_combined:
                self.nuclei_info_data = [_x for _x in self.nuclei_info_data 
                                         if _x.studyId == instance]
                log.info("{!r}: Instance mode activated: {}".format(self, instance))
            else:
                self.nuclei_info_data = [_x for _x in self.nuclei_info_data 
                                         if _x.studyId == instance[0] and _x.maskId == instance[1]]
                log.info("{!r}: Instance mode activated: {}".format(self, instance))
                
        # Logging the number of data found in the directory 
        self.samples_cnt = len(self.nuclei_info_data)
        log.info("{!r}: Mask mode, nuclei combined: {}".format(self, self.nuclei_combined))  
        log.info("{!r}: {} samples".format(self, self.samples_cnt))                       

   
    def shuffle_samples(self):
         # Shuffeling dataset
        random.shuffle(self.nuclei_info_data)

   
    def __len__(self):
         # Function that is "virtual void" type. We need to return number of samples
        return self.samples_cnt
    
    
    def __getitem__(self, ndx):
        # Function that is "virtual void" type. We need to return one sample from the dataset with index
        _nuclei = self.nuclei_info_data[ndx]
        
        _img, _mask = get_nuclei_sample(_nuclei, self.nuclei_combined, self.patch_size)

        _img_t = torch.from_numpy(_img)
        _img_t = _img_t.to(torch.float32)
        _img_t = _img_t.permute(2,0,1)
        _img_t = _img_t[0:3]
        _img_t /= 255.0
        #_img_t = _img_t.unsqueeze(0)
        
        _mask_t = torch.from_numpy(_mask)
        _mask_t = _mask_t.to(torch.float32)
        _mask_t /= 255.0
        _mask_t = _mask_t.unsqueeze(0)
        _mask_t = (_mask_t > 0.5).float()
        #print(_img_t.shape, _mask_t.shape) # fix
        #print(_nuclei) #fix
        #print("\n") #fix
        return (_img_t, _mask_t, _nuclei)
