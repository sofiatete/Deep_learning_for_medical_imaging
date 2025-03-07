import torch.nn as nn
import torch
import torchvision
import pytorch_lightning as pl
import torch.nn.functional as F

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models

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

# Transfer learning with VGG model
class VGG16Classifier(pl.LightningModule):
    def __init__(self, *args):
        super().__init__()
        self.save_hyperparameters()
        self.counter = 0

        # Load pre-trained VGG16
        self.model = models.vgg16(pretrained=True)

        for param in self.model.features.parameters():
            param.requires_grad = False 

        # Modify classifier for binary classification
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_features, 1) 

    def forward(self, X):
        return self.model(X)

class UNet(pl.LightningModule):
  def __init__(self, n_classes=1, in_ch=3):
      super().__init__()
      #######################
      # Start YOUR CODE    #
      #######################
      # number of filter's list for each expanding and respecting contracting layer
      c = [16, 32, 64, 128]

      # first convolution layer receiving the image
      self.enc1 = conv3x3_bn(in_ch, c[0])
      
      # encoder layers
      self.pool1 = nn.MaxPool2d(2)
      self.enc2 = conv3x3_bn(c[0], c[1])

      self.pool2 = nn.MaxPool2d(2)
      self.enc3 = conv3x3_bn(c[1], c[2])

      self.pool3 = nn.MaxPool2d(2)
      self.enc4 = conv3x3_bn(c[2], c[3])

      self.pool4 = nn.MaxPool2d(2)

      # bottleneck
      self.bottleneck = conv3x3_bn(c[3], c[3] * 2)

      # decoder layers
      self.dec4 = deconv(c[3] * 2, c[3])
      self.dec3 = deconv(c[3], c[2])
      self.dec2 = deconv(c[2], c[1])
      self.dec1 = deconv(c[1], c[0])

      # last layer returning the output
      self.final_conv = nn.Conv2d(c[0], n_classes, kernel_size=1)

      #######################
      # END OF YOUR CODE    #
      #######################
  
  def forward(self,x):
      #######################
      # Start YOUR CODE    #
      #######################
      # encoder
      e1 = self.enc1(x)
      p1 = self.pool1(e1)
      e2 = self.enc2(p1)
      p2 = self.pool2(e2)
      e3 = self.enc3(p2)
      p3 = self.pool3(e3)
      e4 = self.enc4(p3)
      p4 = self.pool4(e4)
    
      # bottleneck
      b = self.bottleneck(p4)

      # decoder
      d4 = self.dec4(b, e4)
      d3 = self.dec3(d4, e3)
      d2 = self.dec2(d3, e2)
      d1 = self.dec1(d2, e1)

      # output
      x = self.final_conv(d1)
      #######################
      # END OF YOUR CODE    #
      #######################
      return x



def conv3x3_bn(ci, co):
    #######################
    # Start YOUR CODE    #
    #######################
    return nn.Sequential(
        nn.Conv2d(ci, co, kernel_size=3, padding=1),
        nn.BatchNorm2d(co),
        nn.ReLU(inplace=True),
        nn.Conv2d(co, co, kernel_size=3, padding=1),
        nn.BatchNorm2d(co),
        nn.ReLU(inplace=True),
    )
    #######################
    # end YOUR CODE    #
    #######################

def encoder_conv(ci, co):
    #######################
    # Start YOUR CODE    #
    #######################
    return nn.Sequential(
        conv3x3_bn(ci, co),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )
    #######################
    # end YOUR CODE    #
    #######################

class deconv(nn.Module):
  def __init__(self, ci, co):
    super(deconv, self).__init__()
    #######################
    # Start YOUR CODE    #
    #######################
    self.upconv = nn.ConvTranspose2d(ci, co, kernel_size=2, stride=2)
    self.conv = conv3x3_bn(ci, co)
    #######################
    # end YOUR CODE    #
    #######################

  def forward(self, x1, x2):
      #######################
      # Start YOUR CODE    #
      #######################
      x1 = self.upconv(x1)
      x = torch.cat([x2, x1], dim=1)  # Concatenation along channel dimension
      x = self.conv(x)
      #######################
      # end YOUR CODE    #
      #######################
      return x
