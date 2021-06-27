## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # Image shape (w , h) = (224 X 224) = (n x n) = (w , h, nc) 
        # Calculating the output shape of the convolutional layer
        # output = (input + 2p - f)/s + 1
        # input : image shape ( weight, height)
        # p : Padding
        # f : filters
        # s : stride
        
        self.conv1 = nn.Conv2d(1, 32, 5) # outshape of an image is (1, 220, 220)
        # output = (224 + 2(0) - 5)/1 + 1 = 220
        self.pool1 = nn.MaxPool2d(2, 2)
        # pool = (2,2) = output/2 = 110
        # dropout layer
        self.dropout1 = nn.Dropout(p=0.25)
        
        self.conv2 = nn.Conv2d(32, 64, 3) # output of an image is (32, 110, 110)
        # output shape = (110 + 2(0) - 3)/1 + 1 = 108
        self.pool2 = nn.MaxPool2d(2, 2)
        # pool = (2,2) = output/2 = 108/2 = 54
        # dropout layer
        self.dropout2 = nn.Dropout(p=0.25)
        
        self.conv3 = nn.Conv2d(64, 128, 3) # output of an image is (32, 54, 54)
        # output shape = (54 + 2(0) - 3)/1 + 1 = 52
        self.pool3 = nn.MaxPool2d(2, 2)
        # pool = (2,2) = output/2 = 52/2 = 26
        # dropout layer
        self.dropout3 = nn.Dropout(p=0.25)

        self.conv4 = nn.Conv2d(128, 256, 3) # output of an image is (32, 26, 26)
        # output = (26 + 2(0) - 3)/1 + 1 = 24
        self.pool4 = nn.MaxPool2d(2, 2)
        # pool = (2,2) = output/2 = 24/2 = 12
        # dropout layer
        self.dropout4 = nn.Dropout(p=0.25)

        self.conv5 = nn.Conv2d(256, 512, 1) # output of an image is (32, 12, 12)
        # output = (12 + 2(0) - 3)/1 + 1 = 10
        self.pool5 = nn.MaxPool2d(2, 2)
        # pool = (2,2) = output/2 = 12/2 = 6
        # dropout layer
        
        # fully connected layers
        # output =  (6 + 2(0) - 1)/1 + 1 = 6
        self.fc1 = nn.Linear(512 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, 136) # output of the layer is 136


        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(self.pool1(F.relu(self.conv1(x))))
        x = self.dropout2(self.pool2(F.relu(self.conv2(x))))
        x = self.dropout3(self.pool3(F.relu(self.conv3(x))))
        x = self.dropout4(self.pool4(F.relu(self.conv4(x))))
        x = (self.pool5(F.relu(self.conv5(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
