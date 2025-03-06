import os
import glob
import nibabel as nib
import torch
import numpy as np
from scipy.ndimage import rotate
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torchvision import transforms

from skimage.util import random_noise
from skimage.transform import resize
import random
from PIL import Image


# Data loader
class Scan_DataModule(pl.LightningDataModule):
  def __init__(self, config,transform=True):
    super().__init__()
    self.train_data_dir   = config['train_data_dir']
    self.val_data_dir     = config['val_data_dir']
    self.test_data_dir    = config['test_data_dir']
    self.batch_size       = config['batch_size']
    
    
    if transform:
      self.train_transforms = transforms.Compose([
          Random_Rotate(0.1),
          GaussianNoise(mean=0.0, std=1.0, probability=0.5),
          RandomFlip(horizontal=True, vertical=False, probability=0.5),
          transforms.ToTensor()
      ])
    else:
      self.train_transforms = transforms.Compose([transforms.ToTensor()])
    
    self.val_transforms = transforms.Compose([transforms.ToTensor()])

  def setup(self, stage=None):
    self.train_dataset = Scan_Dataset(self.train_data_dir, transform = self.train_transforms)
    self.val_dataset   = Scan_Dataset(self.val_data_dir  , transform = self.val_transforms)
    self.test_dataset = Scan_Dataset(self.test_data_dir  , transform = self.val_transforms)

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle = False)

  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle = False)


class Scan_DataModule_Segm(pl.LightningDataModule):
  def __init__(self, config, transform=False):
    super().__init__()
    self.train_data_dir   = config['train_data_dir']
    self.val_data_dir     = config['val_data_dir']
    self.test_data_dir    = config['test_data_dir']
    self.batch_size       = config['batch_size']

    if transform:
      self.train_transforms = transforms.Compose([
          Random_Rotate_Seg(0.1),
          GaussianNoise_Seg(mean=0.0, std=1.0, probability=0.5),
          RandomFlip_Seg(horizontal=True, vertical=False, probability=0.5),
          ToTensor_Seg()
      ])
    else:
      self.train_transforms = transforms.Compose([ToTensor_Seg()])

    self.val_transforms = transforms.Compose([ToTensor_Seg()])


  def setup(self, stage=None):
    self.train_dataset = Scan_Dataset_Segm(self.train_data_dir, transform = self.train_transforms)
    self.val_dataset   = Scan_Dataset_Segm(self.val_data_dir  , transform = self.val_transforms)
    self.test_dataset = Scan_Dataset_Segm(self.test_data_dir  , transform = self.val_transforms)

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle=False)

  def test_dataloader(self):
      return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

# Data module
class Scan_Dataset(Dataset):
    def __init__(self, data_dir, transform=False):
        self.transform = transform
        self.data_list = sorted(glob.glob(os.path.join(data_dir, 'img*.nii.gz')))


    def __len__(self):
        """defines the size of the dataset (equal to the length of the data_list)"""
        return len(self.data_list)

    def __getitem__(self, idx):
        """ensures each item in data_list is randomly and uniquely assigned an index (idx) so it can be loaded"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # loading image
        image_name = self.data_list[idx]
        image = nib.load(image_name).get_fdata()
        # image = np.transpose(image, (2, 0, 1))

        # setting label from image name
        label = int(image_name.split('.')[0][-1])
        label = torch.tensor(label)

        # apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


class Scan_Dataset_Segm(Dataset):
  def __init__(self, data_dir, transform=False):
    self.transform = transform
    self.img_list = sorted(glob.glob(os.path.join(data_dir,'img*.nii.gz')))
    self.msk_list = sorted(glob.glob(os.path.join(data_dir,'msk*.nii.gz')))

  def __len__(self):
    """defines the size of the dataset (equal to the length of the data_list)"""
    return len(self.img_list)

  def __getitem__(self, idx):
      """ensures each item in data_list is randomly and uniquely assigned an index (idx) so it can be loaded"""

      if torch.is_tensor(idx):
        idx = idx.tolist()

      # loading image
      image_name = self.img_list[idx]
      image = nib.load(image_name).get_fdata()

      # loading mask
      mask_name = self.msk_list[idx]
      mask = nib.load(mask_name).get_fdata()
      mask = np.expand_dims(mask, axis=2)

      # Convert to PIL Image
      image = Image.fromarray(image.astype(np.uint8))
      mask = Image.fromarray(mask.astype(np.uint8))

      # make sample
      sample = {'image': image, 'mask': mask}

      # apply transforms
      if self.transform:
        sample = self.transform(sample)

      return sample

# data augmentation. You can edit this to add additional augmentation options
class Random_Rotate(object):
  """Rotate ndarrays in sample."""
  def __init__(self, probability):
    assert isinstance(probability, float) and 0 < probability <= 1, 'Probability must be a float number between 0 and 1'
    self.probability = probability

  def __call__(self, sample):
    if float(torch.rand(1, dtype=torch.float64)) < self.probability:
      angle = float(torch.randint(low=-10, high=11, size=(1,)))
      sample = rotate(sample, angle, axes=(0, 1), reshape=False, order=3, mode='nearest')
    return sample.copy()


class Random_Rotate_Seg(object):
  """Rotate ndarrays in sample."""
  def __init__(self, probability):
    assert isinstance(probability, float) and 0 < probability <= 1, 'Probability must be a float number between 0 and 1'
    self.probability = probability

  def __call__(self, sample):
    image, mask = sample['image'], sample['mask']
    if float(torch.rand(1, dtype=torch.float64)) < self.probability:
      angle = float(torch.randint(low=-10, high=11, size=(1,)))
      image = rotate(image, angle, axes=(0, 1), reshape=False, order=3, mode='nearest')
      mask = rotate(mask, angle, axes=(0, 1), reshape=False, order=0, mode='nearest')
    return {'image': image.copy(), 'mask': mask.copy()}


class ToTensor_Seg(object):
  """applies ToTensor for dict input"""
  def __call__(self, sample):
    image, mask = sample['image'], sample['mask']
    image = transforms.ToTensor()(image)
    mask = transforms.ToTensor()(mask)
    return {'image': image.clone(), 'mask': mask.clone()}
  
# Gaussian Noise
class GaussianNoise(object):
    """Efficiently add Gaussian noise to an image."""
    def __init__(self, mean=0.0, std=1.0, probability=0.5):
        assert isinstance(mean, (int, float)), 'Mean must be a number'
        assert isinstance(std, (int, float)) and std > 0, 'Std must be a positive number'
        assert isinstance(probability, float) and 0 < probability <= 1, 'Probability must be a float between 0 and 1'

        self.mean = mean
        self.std = std
        self.probability = probability

    def __call__(self, sample):
        if float(torch.rand(1)) < self.probability:
            # Ensure sample is a NumPy array for consistency
            if isinstance(sample, torch.Tensor):
                sample = sample.numpy()  # Convert tensor to NumPy array if necessary

            # Add Gaussian noise using NumPy for efficiency
            noise = np.random.normal(self.mean, self.std, sample.shape)
            sample = sample + noise  # Element-wise addition

        return sample.copy()  # Ensure it returns a new instance, not a reference
    

class GaussianNoise_Seg(object):
    """Add Gaussian noise to both image and mask in a segmentation task."""
    def __init__(self, mean=0.0, std=1.0, probability=0.5):
        assert isinstance(mean, (int, float)), 'Mean must be a number'
        assert isinstance(std, (int, float)) and std > 0, 'Std must be a positive number'
        assert isinstance(probability, float) and 0 < probability <= 1, 'Probability must be a float between 0 and 1'
        
        self.mean = mean
        self.std = std
        self.probability = probability

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if float(torch.rand(1)) < self.probability:
            # Add Gaussian noise to the image using NumPy
            image = random_noise(image.numpy(), mode='gaussian', mean=self.mean, var=self.std**2)
            image = torch.tensor(image, dtype=torch.float32).clone()

            # Optionally, add noise to the mask if required (uncomment if needed)
            # mask = random_noise(mask.numpy(), mode='gaussian', mean=self.mean, var=self.std**2)
            # mask = torch.tensor(mask, dtype=torch.float32).clone()

        return {'image': image, 'mask': mask.clone()}

# RandomFlip
class RandomFlip(object):
    """Randomly flip an image horizontally and/or vertically."""
    def __init__(self, horizontal=True, vertical=False, probability=0.5):
        assert isinstance(horizontal, bool), 'horizontal must be a boolean value'
        assert isinstance(vertical, bool), 'vertical must be a boolean value'
        assert isinstance(probability, float) and 0 <= probability <= 1, 'Probability must be a float between 0 and 1'
        
        self.horizontal = horizontal
        self.vertical = vertical
        self.probability = probability

    def __call__(self, sample):
        if float(torch.rand(1)) < self.probability:
            # Apply horizontal flip
            if self.horizontal:
                sample = np.flip(sample, axis = 1)  # Flip along width
            # Apply vertical flip
            if self.vertical:
                sample = np.flip(sample, axis = 0)  # Flip along height
        return sample.copy()

class RandomFlip_Seg(object):
    """Randomly flip image and mask horizontally and/or vertically in a segmentation task."""
    def __init__(self, horizontal=True, vertical=False, probability=0.5):
        assert isinstance(horizontal, bool), 'horizontal must be a boolean value'
        assert isinstance(vertical, bool), 'vertical must be a boolean value'
        assert isinstance(probability, float) and 0 <= probability <= 1, 'Probability must be a float between 0 and 1'
        
        self.horizontal = horizontal
        self.vertical = vertical
        self.probability = probability

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        if float(torch.rand(1)) < self.probability:
            # Apply horizontal flip
            if self.horizontal and random.random() < 0.5:
                image = np.flip(image.numpy(), axis=2)  # Flip along width
                mask = np.flip(mask.numpy(), axis=2)  # Flip along width

            # Apply vertical flip
            if self.vertical and random.random() < 0.5:
                image = np.flip(image.numpy(), axis=1)  # Flip along height
                mask = np.flip(mask.numpy(), axis=1)  # Flip along height

        return {'image': torch.tensor(image).float().clone(), 'mask': torch.tensor(mask).float().clone()}
