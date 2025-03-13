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

        # Single skip connection
        self.skip = nn.Conv2d(32, 64, kernel_size=1, stride=1)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU()
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        identity = x  # Save identity for skip connection

        x = self.conv2(x) + self.skip(identity)  # Skip connection
        x = self.pool(x)

        x = self.conv3(x)
        x = self.pool(self.conv4(x))

        x = self.pool(self.conv5(x))

        return self.classifier(x)

# Transfer learning with ResnET Model
class ResNet50(pl.LightningModule):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        
        # Load ResNet-50 Pretrained Model
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Modify First Layer to Handle 3-Channel Images
        self.resnet.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove Fully Connected Layer
        self.resnet.fc = nn.Identity()
        
        # Freeze Early Layers for Transfer Learning
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Freeze entire model initially
        
        # Unfreeze the last few layers (fine-tuning)
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        
        # Unfreeze last residual block
        
        # Custom Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=512),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(in_features=512, out_features=128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x):
        x = self.resnet(x)  # Feature extraction using ResNet
        x = self.classifier(x)  # Classification head
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
      self.enc1 = conv3x3_bn(in_ch, c[0])
      
      # Encoder layers with strided convolutions instead of max pooling
      self.enc2 = conv3x3_bn(c[0], c[1], use_stride=True)
      self.enc3 = conv3x3_bn(c[1], c[2], use_stride=True)
      self.enc4 = conv3x3_bn(c[2], c[3], use_stride=True)

      # Bottleneck
      self.bottleneck = conv3x3_bn(c[3], c[3] * 2)

      # Decoder layers
      self.dec4 = deconv(c[3] * 2, c[3])
      self.dec3 = deconv(c[3], c[2])
      self.dec2 = deconv(c[2], c[1])
      self.dec1 = deconv(c[1], c[0])

      # Output layer
      self.final_conv = nn.Conv2d(c[0], n_classes, kernel_size=1)

      #######################
      # END OF YOUR CODE    #
      #######################
  
  def forward(self,x):
      #######################
      # Start YOUR CODE    #
      #######################
      # Encoder
      e1 = self.enc1(x)
      e2 = self.enc2(e1)
      e3 = self.enc3(e2)
      e4 = self.enc4(e3)

     # Bottleneck
      b = self.bottleneck(e4)

      # Decoder
      d4 = self.dec4(b, e4)
      d3 = self.dec3(d4, e3)
      d2 = self.dec2(d3, e2)
      d1 = self.dec1(d2, e1)

      # Output
      x = self.final_conv(d1)
      #######################
      # END OF YOUR CODE    #
      #######################
      return x



def conv3x3_bn(ci, co, use_stride=False):
    #######################
    # Start YOUR CODE    #
    #######################
    stride = 2 if use_stride else 1
    return nn.Sequential(
        nn.Conv2d(ci, co, kernel_size=3, padding=1, stride=stride),
        nn.BatchNorm2d(co),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Conv2d(co, co, kernel_size=3, padding=1),
        nn.BatchNorm2d(co),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout(0.2),
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
    self.upconv = nn.ConvTranspose2d(ci, co, kernel_size=2, stride=2, padding = 1)
    self.conv = conv3x3_bn(ci, co)
    #######################
    # end YOUR CODE    #
    #######################

  def forward(self, x1, x2):
      #######################
      # Start YOUR CODE    #
      #######################
      x1 = self.upconv(x1)
      if x1.shape[-2:] != x2.shape[-2:]:
          x1 = F.interpolate(x1, size=x2.shape[-2:], mode='bilinear', align_corners=True)
      x = torch.cat([x2, x1], dim=1)  # Concatenation along channel dimension
      x = self.conv(x)
      #######################
      # end YOUR CODE    #
      #######################
      return x
