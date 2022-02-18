from nuclei_dataset import nuclei_dataset
from collections import namedtuple
import torch
from torch.utils.data import DataLoader
from util.logconf import logging
import numpy as np
from model_unet_original import UNet
from augmentation_model import augmentation_model
import datetime
import matplotlib.pyplot as plt

# Logging part
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


# Tuples
model_params = namedtuple(
    'model_params',
    'epochs, batch_size, valid_step, lr, use_gpu',
)

dataset_params = namedtuple(
    'dataset_params',
    'train_path, validation_path, mask, size, n_workers',
)

augmentation_params = namedtuple(
    'augmentation_params',
    'flip, rotate, noise',
)

class nuclei_training_app:
    """Class for training the nuclei models. It handles both: training and validation trough the arguments provided in three tupples.
    
    Args:
        * dataset_parameters: named tuple handeling the dataset Args: 

            - train_path = string, path to train data set,

            - validation_path = string, path to validation dataset,

            - mask = boolean, True for mask combined mode, False for separate masks

            - size = int, size of the input data in mask = True, or patch size in mask = False mode

            - n_workers = int (0,2,4,8), number of subprocesses that are generating dataset

        * augumentation_params: named tuple handeling augmentation. Args: 
            
            - flip: boolean, perform random fliping of the input image over x/y axis

            - rotate: boolean, perform random rotation of the input image

            - noise: float (0 <= x <= 1), random noise summed to the input image

        * model_params: named tuple handeling model training. Args:

            - epochs = int, number of epoch to perform training

            - batch_size = int, batch size

            - valid_step = int, number of epoch after each validation is performed

            - lr = float, learning rate

            - use_gpu = boolean, using gpu (True)

    """
    def __init__(self, dataset_parameters, model_parameters, augmentation_parameters):

        """
            Init function that maps given arguments and enables GPU
        """

        self.lr = model_parameters.lr
        self.batch_size = model_parameters.batch_size
        self.gpu = model_parameters.use_gpu
        self.epochs = model_parameters.epochs
        self.valid_eval_step = model_parameters.valid_step
        
        self.data_path_train = dataset_parameters.train_path
        self.data_path_valid = dataset_parameters.validation_path
        self.nuclei_combined = dataset_parameters.mask
        self.number_of_workers = dataset_parameters.n_workers
        self.size = dataset_parameters.size
        self.aug = augmentation_parameters
        
        self.use_cuda = torch.cuda.is_available()        
        self.device = torch.device("cuda" if self.use_cuda and self.gpu else "cpu")
        self.loss = torch.nn.BCELoss(reduction = 'none')
        
        self.model, self.aug_model = self.init_models()
        self.optimizer = self.init_optimizer()

    def init_models(self):
        """
        Model init. UPGRADE TO INIT MODEL WEIGHTS
        """
        # Init models
        _model = UNet(padding = 1)
        _aug_model = augmentation_model(self.aug)
        
        # Place them on GPU
        if self.gpu:
            log.info(f"Using cuda: {torch.cuda.get_device_name(self.device)}")
            
            if torch.cuda.device_count() > 1:
                _model = torch.nn.DataParallel(_model)
                _aug_model = torch.nn.DataParallel(_aug_model)
            
            _model = _model.to(self.device)
            _aug_model = _aug_model.to(self.device)
        return _model, _aug_model
    
    
    def init_optimizer(self):
        """
            Init optimizer: Feel free to add other optmizers. UPGRADE: optimizer as param
        """
        return torch.optim.Adam(self.model.parameters(), lr = self.lr)
        #return torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.99)
    
    def init_train_dataloader(self):
        """
            Init of the train data loader. NOT TESTED FOR MULTIPLE GPU
            Creating wrapper arround data class. 
        """
        _train_ds = nuclei_dataset(
            nuclei_dir_path = self.data_path_train,
            nuclei_combined = self.nuclei_combined,
            patch_size = self.size
        )
        if self.use_cuda and self.gpu:
            self.batch_size *= torch.cuda.device_count()

        _train_dl = DataLoader(
            _train_ds,
            batch_size=self.batch_size,
            num_workers=self.number_of_workers,
            pin_memory=self.use_cuda,
        )  
        return _train_dl
    
    def init_val_dataloader(self): 
        """
            Init of the validation data loader. NOT TESTED FOR MULTIPLE GPU
            Creating wrapper arround data class. 
        """      
        _valid_ds = nuclei_dataset(
            nuclei_dir_path = self.data_path_valid,
            nuclei_combined = self.nuclei_combined,
            patch_size = self.size
        ) 

        if self.use_cuda and self.gpu:
            self.batch_size *= torch.cuda.device_count()

        _valid_dl = DataLoader(
            _valid_ds,
            batch_size=self.batch_size,
            num_workers=self.number_of_workers,
            pin_memory=self.use_cuda,
        )

        return _valid_dl
    

    def train_model(self, data):
        """
        Training model function. 

        Args:
            * data, dataloader of the train dataset
        """

        # Metrics
        # Loss TP FP FN 
        _metrics = torch.zeros(4, len(data.dataset), device = self.device)
        
        # Swap to mode train
        self.model.train()

        # Shuffle dataset and create enum object
        data.dataset.shuffle_samples()
        _batch_iter = enumerate(data)
        
        # Go trough batches
        for _index, _batch in _batch_iter:
            # Clear grads
            self.optimizer.zero_grad()
            
            # Calc loss
            _loss = self.get_loss(_index, _batch, _metrics)
            
            # Propagate loss
            _loss.backward()
            
            # Apply loss
            self.optimizer.step()
        
        # Return metrics
        return _metrics
    
    def validate_model(self, data):
        """
        Validation model function

        Args:
            * data, dataloader of the train dataset
        """

        # Metrics
        # Loss TP FP FN 
        _metrics = torch.zeros(4, len(data.dataset), device = self.device)

        # We don't need calculate gradients 
        with torch.no_grad():
            # Set model in evaluate mode - no batchnorm and dropout
            self.model.eval()

            # Go trough data
            for _index, _batch in enumerate(data):
                # Get loss
                _loss = self.get_loss(_index, _batch, _metrics, aug = False)
        
        # Return metrics        
        return _metrics
    
    
    def get_loss(self, _index, _batch, _metrics, aug = True):
        """
        Function that calculates loss. Loss in this code is BinaryCrossEntropy

        Args:
            * _index, int, batch index needed to populate _metrics

            * _batch, tensor, data

            * _metrics, tensor, container to save data
        """

        # Parse _batch
        _input_data, _mask_data, _info_data = _batch
        
        # Augmenet data
        if aug:
            _input_data, _mask_data = self.aug_model(_input_data, _mask_data)
        
        # Transfer data
        _input_data = _input_data.to(self.device, non_blocking = True)
        _mask_data = _mask_data.to(self.device, non_blocking = True)
        
        
        # Loss
        # Caluclate loss
        _prediction = self.model(_input_data)
        _loss = self.loss(_prediction, _mask_data)
        
        # For metrics
        _begin_index = self.batch_size * _index
        _end_index = _begin_index + _input_data.size(0)
        
        # Calculate metrics
        _classification_threshold = 0.5
        with torch.no_grad():
            _prediction_bool = (_prediction[: , 0:1] > 
                                _classification_threshold).to(torch.float32)
            _tp = (_prediction_bool * _mask_data.to(torch.bool)).sum(dim=[1,2,3])
            _fn = ((1-_prediction_bool) * _mask_data.to(torch.bool)).sum(dim=[1,2,3])
            _fp = (_prediction_bool * ~_mask_data.to(torch.bool)).sum(dim=[1,2,3])
            
            _metrics[0, _begin_index:_end_index] = torch.mean(_loss, dim = (1,2,3))
            _metrics[1, _begin_index:_end_index] = _tp
            _metrics[2, _begin_index:_end_index] = _fp
            _metrics[3, _begin_index:_end_index] = _fn

        # Return mean of all loss          
        return _loss.mean()

    def eval_metrics(self, epoch, metrics, mode):
        """
            Function for metric evaluatio-calculating recall, precission and f1_score
        """
        # Loss TP FP FN 
        _metrics = metrics.to('cpu')
        _metrics = _metrics.detach().numpy()
        _metrics_sum = _metrics.sum(axis=1)
        _metrics_dict = {}
        
        _metrics_dict['loss'] = _metrics[0].mean()
        _metrics_dict['tp'] = _metrics_sum[1] \
            / ((_metrics_sum[1] + _metrics_sum[3]) or 1) * 100
        _metrics_dict['fp'] = _metrics_sum[2] \
            / ((_metrics_sum[1] + _metrics_sum[3]) or 1) * 100
        _metrics_dict['fn'] = _metrics_sum[3] \
            / ((_metrics_sum[1] + _metrics_sum[3]) or 1) * 100
        
        _metrics_dict['precision'] = _metrics_dict['tp'] \
            / ((_metrics_dict['tp'] + _metrics_dict['fp']) or 1)
        _metrics_dict['recall'] = _metrics_dict['tp'] \
            / ((_metrics_dict['tp'] + _metrics_dict['fn']) or 1)
        
        _metrics_dict['f1_score'] = 2 * _metrics_dict['precision'] * _metrics_dict['recall'] \
            / ((_metrics_dict['precision'] + _metrics_dict['recall']) or 1)
        
        log.info(("Epoch{} {:8} "
             + "{loss:.4f} loss, "
             + "{precision:.4f} precision, "
             + "{recall:.4f} recall, "
             + "{f1_score:.4f} f1 score"
              ).format(epoch, mode, **_metrics_dict))
        
        return _metrics_dict['f1_score']

    def load_model(self, path):
        """
            Function that loads model.

            Args:
                * path, string, path to the model checkpoint
        """
        log.info("LOADING MODEL")
        _state_dict = torch.load(path)
        self.model.load_state_dict(_state_dict['model_state'])
        self.optimizer.load_state_dict(_state_dict['optimizer_state'])
        self.optimizer.name = _state_dict['optimizer_name']
        self.model.name = _state_dict['optimizer_name']
        log.info(f"LOADING MODEL, epoch {_state_dict['epoch']}"
                 + f", time {_state_dict['time']}")
       
    def save_model(self, epoch, best):
        """
            Function for model saving

            Args:
                * epoch, int, epoch being saved

                * best, boolean, Is this the best model
        """

        _model = self.model
        if isinstance(_model, torch.nn.DataParallel):
            _model = _model.module
        
        _state = {
            'time': str(datetime.datetime.now()),
            'model_state': _model.state_dict(),
            'model_name': type(_model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch
        }
        torch.save(_state, 'last_model-true.pth')
        log.info('Saving model!')
        
        if best:
            log.info('Saving best model!')
            torch.save(_state, 'best_model-true.pth')

    def eval_model(self, sample):
        """
            Function to test model on one sample

            Args:
                *sample, tensor, image+mask+info
        """

        _sample = nuclei_dataset(self.data_path_train, 
                                  instance = sample,
                                 nuclei_combined = self.nuclei_combined,
                                 patch_size = self.size)
        _input_img, _real_mask, _info = _sample[0]
        with torch.no_grad():
            self.model.eval()
            _input_img = _input_img.unsqueeze(0).to(self.device)
            _prediction = self.model(_input_img)
        
        _input_img = _input_img.to('cpu')
        _prediction = _prediction.to('cpu')
        _real_mask = _real_mask.to('cpu')
        
        self.plot_sample(_input_img.squeeze(0), _real_mask, _prediction.squeeze(0))
    
    def plot_sample(self, input_img, mask, predicted):
        """
            Function to plot sample
        """

        f, _axarr = plt.subplots(1,3, figsize=(50, 50)) 

        # use the created array to output your multiple images. In this case I have stacked 4 images vertically
        _axarr[0].imshow(input_img.permute(1,2,0))
        _axarr[0].set_title('Input image')
        _axarr[1].imshow(mask.permute(1,2,0), cmap='gray')
        _axarr[1].set_title('Target mask')
        _axarr[2].imshow(predicted.permute(1,2,0), cmap='gray')
        _axarr[2].set_title('Predicted mask')
        plt.show()          
    
    def main(self):
        """
            Main train function.
        """
        log.info(f"Starting training {self}")

        # Get datasets
        _train_dl = self.init_train_dataloader()
        _valid_dl = self.init_val_dataloader()
            
            
        # Set score 
        _best_score = 0
        _best_epoch = 0
        for _epoch in range(1, self.epochs +1):
            log.info(f"Epoch {_epoch} / {self.epochs}")
            
            # Trening
            _metrics = self.train_model(_train_dl)
            self.eval_metrics(_epoch, _metrics, 'Train')
            self.save_model(_epoch, best = False)
            
            # Validation
            if _epoch == 1 or _epoch % self.valid_eval_step == 0:
                _metrics = self.validate_model(_valid_dl)
                
                _f1_score = self.eval_metrics(_epoch, _metrics, 'Valid')
                
                if _epoch == 1 or (_f1_score > _best_score):
                    self.save_model(_epoch, best = True)
                    _best_score = _f1_score
                    _best_epoch = _epoch
            
            # Early stopping
            if _epoch - _best_epoch  > self.valid_eval_step * 3:
                log.info(f"Early stopping at epoch: {_epoch}")
                break

