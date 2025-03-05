import torch.nn as nn
import torch
import torchvision
import pytorch_lightning as pl
import torch.nn.functional as F


class SimpleConvNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Initial convolutional block: sets the foundation for skip connections
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16,kernel_size=5, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # List of convolutional blocks with increasing channels and skip connections
        self.conv_blocks = nn.ModuleList()
        in_channels = 16
        out_channels = 32

        # Creating 5 convolutional blocks with skip connections
        for _ in range(5):
            self.conv_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            ))
            in_channels = out_channels
            out_channels *= 2 if out_channels < 128 else 128

        # Classifier: reduces spatial dimensions and maps features to output
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(4, 4)),
            nn.Flatten(),
            nn.Linear(in_features=4 * 4 * 128, out_features=60),
            nn.ReLU(),
            nn.Linear(in_features=60, out_features=1)
        )
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        # Pass input through initial convolution
        x = self.initial_conv(x)
        skip = x

        # Pass through each convolutional block with skip connection
        for block in self.conv_blocks:
            x = block(x)
            x += skip  # Adding skip connection to preserve information
            skip = x

        # Final classification layer
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
