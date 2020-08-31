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

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])



# TODO: Load the datasets with ImageFolder
trainset = datasets.ImageFolder(train_dir, transform=data_transforms)
testset = datasets.ImageFolder(test_dir, transform=test_transforms)
validset = datasets.ImageFolder(valid_dir, transform=valid_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle =True)

### Label mapping

You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

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
#images = images.view(images.shape[0], -1)

# define loss
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

if torch.cuda.is_available():
    print('cuda is available')
else:
    print('cuda unavailable')

# Train the classifier layers using backpropagation using the pre-trained network to get the features

model.cuda()
epochs = 3
steps = 0
print_every = 60
running_loss = 0
#training loop
for epoch in range(epochs):
    for images, labels in dataloaders:
        running_loss = 0
        steps += 1
        images = images.cuda()
        labels = labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()
        
        logps = model(images)
        loss = criterion(logps, labels)
        loss.backward
        optimizer.step()
        
        # keep track of training loss as we go through data
        running_loss += loss.item()
        # validation
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            # turn off gradients
            with torch.no_grad():
                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)
                    validps = model.forward(images)
                    batch_loss = criterion(validps, labels)
                    valid_loss += batch_loss.item()
                    # accuracy
                    ps = torch.exp(validps)
                    top_prob, top_label = ps.topk(1, dim=1)
                    equals = top_label == labels.view(*top_label.shape)

                    accuracy += torch.mean(equals.type(torch.cuda.FloatTensor)).item()
                    # epoch += 1
                    print("Epoch: {}/{}:".format(epoch + 1, epochs),
                            f"Training_loss: {running_loss/print_every} "
                            f"Validation loss: {valid_loss/len(validloader):.3f} "
                            f"Validation Accuracy: {accuracy/len(validloader):.3f}% ")
                    #running_loss = 0
                    model.train()
        
print('finished training')

# TODO: Do validation on the test set
model.eval()
test_var = 0
accuracy = 0

for data in testloader:
    images, labels = data
    test_var += 1
    images, labels = images.to(device), labels.to(device)
    output = model.forward(images)
    ps = torch.exp(output).data
    #batch_loss = criterion(testps, labels)
    #test_loss += batch_loss.item()
    
    # accuracy
    #ps = torch.exp(testps)
    #top_prob, top_label = ps.topk(1, dim=1)
    equals = (labels.data == ps.max(1)[1])
    accuracy += equals.type_as(torch.FloatTensor()).mean()
#    print(epoch,
#          f"Train Loss: {running_loss/len(dataloaders)}",
#          f"Test Loss: {test_loss/len(testloader)}:.3f",
#          f"Test Accuracy: {accuracy/len(testloader)}:.3f")
print('testing accuracy: {:.4f}'.format(accuracy/test_var))