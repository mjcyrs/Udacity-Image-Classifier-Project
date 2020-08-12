%matplotlib inline
%config InLineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
from PIL import Image
image_path = 'flowers/train/1/image_06734.jpg'
# load image data.
image = Image.open(image_path)
data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Resize(224),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
image_datasets = datasets.ImageFolder(data_dir + '/train', transform=data_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = torch.utils.data.DataLoader(data_transforms, batch_size = 64, shuffle=True)

import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

    # build and train your network here
import torchvision.models as models
model = models.vgg16(pretrained=True)
# freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
model

# feedforward
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                           ('fc1', nn.Linear(1024, 256)),
                           ('relu', nn.ReLU()),
                           ('dropout1', nn.Dropout(p=0.5)),
                           ('fc2', nn.Linear(256, 1)),
                           ('output', nn.LogSoftmax(dim=1))
]))
model.classifier = classifier
# define loss
criterion = nn.CrossEntropyLoss()

# Train the classifier layers using backpropagation using the pre-trained network to get the features
optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer.zero_grad()