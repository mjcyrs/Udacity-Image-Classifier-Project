%matplotlib inline
%config InLineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os
 

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
from PIL import Image
image_path = 'flowers/train/1/image_06734.jpg'
image = Image.open(image_path) # load image data.

# define transform to normalize data
data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.Resize(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
trainset = datasets.ImageFolder(train_dir, transform=data_transforms)
# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle=False)


import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)



# build and train your network here

import torchvision.models as models

# Get our data
images, labels = next(iter(dataloaders))

model = models.vgg16(pretrained=True)
model
# freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = True

# feedforward

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                           ('fc1', nn.Linear(25088, 4096)),
                           ('relu', nn.ReLU()),
                           ('dropout1', nn.Dropout(p=0.5)),
                           ('fc2', nn.Linear(4096, 102)),
                           ('output', nn.LogSoftmax(dim=1))
]))

model.classifier = classifier
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to = device


# Flatten images
images = images.view(images.shape[0], -1)

# define loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

if torch.cuda.is_available():
    print('cuda is available')
else:
    print('cuda unavailable')
    
#cpu = torch.device('cpu')


# Train the classifier layers using backpropagation using the pre-trained network to get the features

model.cuda()
print("we are there, man")
#for device in ['cpu', 'cuda']:
epochs = 2

running_loss = 0.0
accuracy = 0
#training loop
for epoch in range(1, epochs):
    images = images.cuda()
    labels = labels.cuda()
    for images, labels in dataloaders:
        print('epoch number: {}'.format(epoch))
        # zero the parameter gradients
        optimizer.zero_grad()
       

        # forward + backward + optimize
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # update statistics
        running_loss += loss.item()
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(dataloaders)))
    print(f'Accuracy: {accuracy*100}%')
    print('Finished Training')
