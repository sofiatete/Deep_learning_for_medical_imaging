import torch.nn as nn
import torch
import torchvision
import pytorch_lightning as pl
import torch.nn.functional as F


import torch
import torch.nn as nn
import pytorch_lightning as pl

class SimpleConvNet(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, padding=3), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.skip1 = nn.Conv2d(3, 64, kernel_size=1, stride=1)  # Skip connection 1
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.skip2 = nn.Conv2d(64, 256, kernel_size=1, stride=1)  # Skip connection 2
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU()
        )
        self.skip3 = nn.Conv2d(256, 512, kernel_size=1, stride=1)  # Skip connection 3
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU()
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU()
        )
        self.skip4 = nn.Conv2d(512, 512, kernel_size=1, stride=1)  # Skip connection 4
        
        self.pool = nn.MaxPool2d(2, 2)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x) + self.skip1(identity)  # Skip connection 1
        x = self.pool(x)

        identity = x
        x = self.conv3(x)
        x = self.conv4(x) + self.skip2(identity)  # Skip connection 2
        x = self.pool(x)

        identity = x
        x = self.conv5(x)
        x = self.conv6(x) + self.skip3(identity)  # Skip connection 3
        x = self.pool(x)

        identity = x
        x = self.conv7(x)
        x = self.conv8(x) + self.skip4(identity)  # Skip connection 4
        x = self.pool(x)
        return self.classifier(x)

class UNet(pl.LightningModule):
  def __init__(self, n_classes=1, in_ch=3):
      super().__init__()
      #######################
      # Start YOUR CODE    #
      #######################
      # number of filter's list for each expanding and respecting contracting layer
      c = [16, 32, 64, 128]

      # first convolution layer receiving the image
      # encoder layers

      # decoder layers

      # last layer returning the output
      #######################
      # END OF YOUR CODE    #
      #######################
  def forward(self,x):
      #######################
      # Start YOUR CODE    #
      #######################
      # encoder

      # decoder

      #######################
      # END OF YOUR CODE    #
      #######################
      return x


def conv3x3_bn(ci, co):
    #######################
    # Start YOUR CODE    #
    #######################
    pass
    #######################
    # end YOUR CODE    #
    #######################

def encoder_conv(ci, co):
    #######################
    # Start YOUR CODE    #
    #######################
    pass
    #######################
    # end YOUR CODE    #
    #######################

class deconv(nn.Module):
  def __init__(self, ci, co):
    super(deconv, self).__init__()
    #######################
    # Start YOUR CODE    #
    #######################
    pass
    #######################
    # end YOUR CODE    #
    #######################

  def forward(self, x1, x2):
      #######################
      # Start YOUR CODE    #
      #######################
      x=x1
      #######################
      # end YOUR CODE    #
      #######################
      return x
