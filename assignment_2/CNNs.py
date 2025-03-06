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
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # 1x1 Conv for matching dimensions in skip connections
        self.skip1 = nn.Conv2d(3, 16, kernel_size=1, stride=2)
        self.skip2 = nn.Conv2d(16, 32, kernel_size=1, stride=2)
        self.skip3 = nn.Conv2d(32, 64, kernel_size=1, stride=2)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(2 * 2 * 128, 120),
            nn.ReLU(),
            nn.Linear(120, 60),
            nn.ReLU(),
            nn.Linear(60, 1)
        )

    def forward(self, x):
        # First block
        print("foward block 1!")
        identity = self.skip1(x)
        x = self.conv1(x) + identity

        # Second block
        print("foward block 2!")
        identity = self.skip2(x)
        x = self.conv2(x) + identity

        # Third block
        print("foward block 3!")
        identity = self.skip3(x)
        x = self.conv3(x) + identity

        # Fourth block (no skip connection needed as it's the last)
        print("foward block 4!")
        x = self.conv4(x)

        # Classifier
        print("foward classifier!")
        x = self.classifier(x)
        return x

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
