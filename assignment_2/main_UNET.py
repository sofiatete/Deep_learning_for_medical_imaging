"""
Author: adopted from Ricardo Eugenio González Valenzuela by Oliver Gurney-Champion | Spring 2023
Date modified: March 2023

Skin cancer is the most common cancer globally, with melanoma being the most deadly form. Dermoscopy is a skin imaging
modality that has demonstrated improvement for diagnosis of skin cancer compared to unaided visual inspection. However,
clinicians should receive adequate training for those improvements to be realized. In order to make expertise more
widely available, the International Skin Imaging Collaboration (ISIC) has developed the ISIC Archive, an international
repository of dermoscopic images, for both the purposes of clinical training, and for supporting technical research
toward automated algorithmic analysis by hosting the ISIC Challenges.

The dataset is already pre-processed for you in batches of 128x256x256x3 (128 images, of size 256x256 rgb)
"""

#loading libraries

import os
import numpy as np
import argparse
import glob
import pandas as pd

import torch
import torchmetrics
import torch.nn.functional as F
import torch.nn as nn

from torchvision import transforms
from sys import platform
from Data_loader import Scan_Dataset_Segm, Scan_DataModule_Segm, ToTensor_Seg, Random_Rotate_Seg, Random_Flip_Seg, Random_GaussianBlur_Seg
from visualization import show_data_Unet, show_data_logger_Unet
from CNNs import UNet
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import nibabel as nib
import torchio
import matplotlib.pyplot as plt


#start interactieve sessie om wandb.login te runnen
wandb.login()

#seeding for reproducible results
SEED=42
#torch.backends.cudnn.deterministic = True
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
np.random.RandomState(SEED)

if platform == "linux" or platform == "linux2":
    data_dir = '/gpfs/work5/0/prjs1312/data/segmentation'
else:
    #set data location on your local computer. Data can be downloaded from:
    # https://surfdrive.surf.nl/files/index.php/s/epjCz4fip1pkWN7
    # PW: deeplearningformedicalimaging
    data_dir = '/Users/costa/Desktop/Computational_Science/Deep_Learning/ass_2/segmentation'

print('data is loaded from ' + data_dir)
# view data
nn_set = 'train' # ['train', 'val', 'test']
index = 0
dataset = Scan_Dataset_Segm(os.path.join(data_dir, nn_set))
show_data_Unet(dataset,index,n_images_display=5)
train_transforms = transforms.Compose([Random_Rotate_Seg(0.1), Random_Flip_Seg(0.3), Random_GaussianBlur_Seg(),
                                       ToTensor_Seg()])
dataset = Scan_Dataset_Segm(os.path.join(data_dir, nn_set),transform = train_transforms)
show_data_Unet(dataset,index,n_images_display=5)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.device(device)
print('device is ' + device)


models = {'unet': UNet}

optimizers = {'adam': torch.optim.Adam,
              'sgd': torch.optim.SGD}

metrics = {'acc': torchmetrics.Accuracy(task='binary').to(device),
           'f1': torchmetrics.F1Score(task='binary').to(device),
           'precision': torchmetrics.Precision(task='binary').to(device),
           'recall': torchmetrics.Recall(task='binary').to(device)}

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)  # Ensure values are between 0-1
        intersection = torch.sum(y_pred * y_true)
        dice = (2. * intersection + self.smooth) / (torch.sum(y_pred) + torch.sum(y_true) + self.smooth)
        return 1 - dice  # Dice loss is (1 - Dice coefficient)

class Segmenter(pl.LightningModule):
    def __init__(self, *args):
        super().__init__()
        self.save_hyperparameters()
        self.counter=0

        # defining model
        self.model_name = config_segm['model_name']
        assert self.model_name in models, f'Model name "{self.model_name}" is not available. List of available names: {list(models.keys())}'
        self.model = models[self.model_name]().to(device)

        # assigning optimizer values
        self.optimizer_name = config_segm['optimizer_name']
        self.lr = config_segm['optimizer_lr']

        self.criterion = DiceLoss()

    def step(self, batch, nn_set):
        X, y = batch['image'], batch['mask']
        X, y = X.float().to(device), y.to(device).float()
        y_hat = self.model(X)

        y_prob = torch.sigmoid(y_hat)
        self.y_prob=y_prob>0.5
        
        # del X, y_hat, batch

        #pos_weight = torch.tensor([fig_segm['loss_pos_weight']]).float().to(device)
        # loss = F.binary_cross_entropy(y_prob,y)
        loss = self.criterion(y_hat, y)
        self.log(f"{nn_set}_loss", loss, on_step=False, on_epoch=True)

        for i, (metric_name, metric_fn) in enumerate(metrics.items()):
            score = metric_fn(y_hat, y.int())
            self.log(f'{nn_set}_{metric_name}', score, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, 'train')
        if batch_idx == 0:
            fig = show_data_logger_Unet(batch,0,self.y_prob,n_images_display = 5, message = 'train_example')
            self.logger.log_image("train_example",[fig],step=self.counter)
        batch_dictionary = {'loss': loss}
        self.log_dict(batch_dictionary)
        return batch_dictionary


    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, 'val')
        if batch_idx == 0:
            fig = show_data_logger_Unet(batch,0,self.y_prob,n_images_display = 5, message = 'val_example')
            self.logger.log_image("val_example",[fig],step=self.counter)
            self.counter = self.counter+1
        batch_dictionary = {'loss': loss}
        self.log_dict(batch_dictionary)


    def test_step(self, batch, batch_idx):
        self.step(batch, 'test')

    def forward(self, X):
        return self.model(X)

    def configure_optimizers(self):
        assert self.optimizer_name in optimizers, f'Optimizer name "{self.optimizer_name}" is not available. List of available names: {list(models.keys())}'
        return optimizers[self.optimizer_name](self.parameters(), lr=self.lr)


def run(config_segm):
    logger = WandbLogger(name=config_segm['experiment_name'], project='ISIC-Unet')
    data = Scan_DataModule_Segm(config_segm)
    segmenter = Segmenter(config_segm)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath='/gpfs/work5/0/prjs1312/CheckPoints', monitor='val_f1')
    trainer = pl.Trainer(max_epochs=config_segm['max_epochs'],
                         logger=logger, callbacks=[checkpoint_callback],
                         default_root_dir=config_segm['bin'],
                         log_every_n_steps=1)
    trainer.fit(segmenter, data)
    
    # change these paths
    test_data_dir = os.path.join(data_dir, 'test')
    # load best model
    PATH = glob.glob(os.path.join(config_segm['checkpoint_dir'], '*'))[0]
    model = Segmenter.load_from_checkpoint(PATH)
    model.eval()

    # make test dataloader
    test_data = Scan_DataModule_Segm(test_data_dir)

    # test model
    trainer = pl.Trainer()
    trainer.test(model, dataloaders=test_data, verbose=True)

    # get and store predictions
    if config_segm['checkpoint_dir']:
        image_list = glob.glob(test_data_dir+'/img*.nii.gz')
        predictions_dir = config_segm['checkpoint_dir']
        os.makedirs(predictions_dir, exist_ok=True)
        for image in image_list:
            X = torch.tensor(nib.load(image).get_fdata())
            X = torch.permute(X, [2, 0, 1])
            X = torch.unsqueeze(X, 0)
            X = X.float()
            y = model(X)
            pred = torch.sigmoid(y)
            pred_image = torchio.Image(tensor=pred.cpu().detach())
            pred_path = predictions_dir + 'pred_' + image.rsplit('/', 1)[1]
            pred_image.save(pred_path)


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    # Optimizer hyperparameters
    parser.add_argument('--optimizer_lr', default=0.01, type=float, nargs='+',
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Minibatch size')
    parser.add_argument('--model_name', default='unet', type=str,
                        help='defines model to use')
    parser.add_argument('--optimizer_name', default='adam', type=str,
                        help='optimizer options: adam and sgd (default)')
    # Other hyperparameters
    parser.add_argument('--max_epochs', default=50, type=int,
                        help='Max number of epochs')
    parser.add_argument('--experiment_name', default='test', type=str,
                        help='name of experiment')
    parser.add_argument('--checkpoint_folder_save', default=None, type=str,
                        help='path of experiment')

    args = parser.parse_args()
    config_segm = vars(args)

    config_segm.update({
        'train_data_dir': os.path.join(data_dir, 'train'),
        'val_data_dir': os.path.join(data_dir, 'val'),
        'test_data_dir': os.path.join(data_dir, 'test'),
        'bin': 'segm_models/',
        'loss_pos_weight': 1,
        'checkpoint_folder_save': '/gpfs/work5/0/prjs1312/CheckPoints',
    })

    # run(config_segm)

def tune_hyperparameters(epochs_list, lr_list, optimizers_list):
    best_model = None
    best_score = 0  # Track the best validation F1-score
    best_config = {}

    for epochs in epochs_list:
        for lr in lr_list:
            for optimizer_name in optimizers_list:
                print(f"Training with epochs={epochs}, lr={lr}, optimizer={optimizer_name}")

                config = {
                    'max_epochs': epochs,
                    'optimizer_lr': lr,
                    'optimizer_name': optimizer_name,
                    'batch_size': 16, 
                    'model_name': 'unet',
                    'experiment_name': f"exp_{epochs}_{lr}_{optimizer_name}",
                    'train_data_dir': os.path.join(data_dir, 'train'),
                    'val_data_dir': os.path.join(data_dir, 'val'),
                    'test_data_dir': os.path.join(data_dir, 'test'),
                    'bin': 'segm_models/',
                    'loss_pos_weight': 1,
                    'checkpoint_folder_save': '/gpfs/work5/0/prjs1312/CheckPoints',
                }

                try:
                    # Train model
                    run(config)

                    # Ensure each experiment gets a unique log directory
                    log_dir = f"segm_models/lightning_logs/{config['experiment_name']}/metrics.csv"
                    
                    if os.path.exists(log_dir):
                        df = pd.read_csv(log_dir)
                        val_f1 = df[df["val_f1"].notna()]["val_f1"].max()

                        if val_f1 > best_score:
                            best_score = val_f1
                            best_model = config
                            best_config = config.copy()
                    else:
                        print(f"Warning: Metrics file not found for {config['experiment_name']}")

                except Exception as e:
                    print(f"Error during training with {config}: {e}")
                    continue  # Move to the next set of parameters

    print(f"Best Model: {best_config} with F1-score: {best_score}")
    return best_model

# Run hyperparameter tuning
""" best_model = tune_hyperparameters(
    epochs_list=[30, 50],  # Try different epoch numbers
    lr_list=[0.001, 0.01, 0.1],  # Try different learning rates
    optimizers_list=['adam', 'sgd']  # Try Adam and SGD optimizers
) """
