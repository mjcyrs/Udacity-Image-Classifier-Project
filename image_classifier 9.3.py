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
data_transforms = transforms.Compose([transforms.RandomRotation(10),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),                                                                             transforms.ToTensor(),
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
    param.requires_grad = False


# feedforward

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                           ('fc1', nn.Linear(25088, 512)),
                           ('relu', nn.ReLU()),
                           ('dropout1', nn.Dropout(p=0.2)),
                           ('fc2', nn.Linear(512, 102)),
                           ('output', nn.LogSoftmax(dim=1))
]))

model.classifier = classifier
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to = device

# define loss
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.01)

if torch.cuda.is_available():
    print('cuda is available')
else:
    print('cuda unavailable')



# Train the classifier layers using backpropagation using the pre-trained network to get the features
model.cuda()
epochs = 10
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
        loss.backward()
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
                print("Epoch: {}/{}:".format(epoch + 1, epochs),
                            f"Training_loss: {running_loss/print_every} "
                            f"Validation loss: {valid_loss/len(validloader):.3f} "
                            f"Validation Accuracy: {accuracy/len(validloader):.3f}")
                
        
print('finished training')


model.eval()
test_var = 0
accuracy = 0

for data in testloader:
    images, labels = data
    test_var += 1
    images, labels = images.to(device), labels.to(device)
    output = model.forward(images)
    ps = torch.exp(output).data
    
    # accuracy
    equals = (labels.data == ps.max(1)[1])
    accuracy += equals.type_as(torch.FloatTensor()).mean()
print('testing accuracy: {:.4f}%'.format((accuracy/test_var)*100))




# TODO: Save the checkpoint (state dict)s
model.class_to_idx = trainset.class_to_idx
torch.save(model.state_dict(), 'bigcatmodel.pth')
state_dict = torch.load('bigcatmodel.pth')
model.load_state_dict(state_dict)
# rebuild model with information about architecture
checkpoint = {'input_size': 25088,
              'hidden_layer_size': 512,
              'output_size': 102,
              'state_dict': model.state_dict()}
torch.save(checkpoint, 'bigcatmodel.pth')


# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = classifier
    return model


import numpy as np

def process_image(new_image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''    
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(new_image)
    img_size = (256, 256)
    im.thumbnail(img_size)
    width, height = im.size
    new_width, new_height = 224, 224
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    im.crop((left, top, right, bottom))
    
    # convert to numpy array
    np_image = np.array(im)
    
    np_image.astype('float64')
    np_image = np_image / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    
    np_image = np_image.transpose((2, 0, 1))
    return torch.tensor(np_image)
    
processed_image = process_image('flowers/train/1/image_06734.jpg') 



def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))   
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean   
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)    
    ax.imshow(image)   
    return ax
imshow(processed_image)


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''    
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path).type(torch.FloatTensor).unsqueeze_(0).to(device)
    
    load_checkpoint('bigcatmodel.pth')
    
    #model.idx_to_class = dict(map(reversed, model.class_to_idx.items()))
    
    with torch.no_grad():
        outputs = model.forward(image)
        ps = torch.exp(outputs)
        probs, indices = ps.topk(topk)
        probs = probs.squeeze()
        classes = [model.idx_to_class[idx] for idx in indices[0].tolist()]
    
    return probs, classes
#new_image_path = 'flowers/test/1/image_06743.jpg'
path = test_dir + '/1/image_06743.jpg'
predict(path, model)


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0)) 
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean   
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)  
    ax.imshow(image)    
    return ax
imshow(processed_image)


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path).type(torch.FloatTensor).unsqueeze_(0).to(device)
    
    load_checkpoint('bigcatmodel.pth')
    
    #model.idx_to_class = dict(map(reversed, model.class_to_idx.items()))
    
    with torch.no_grad():
        outputs = model.forward(image)
        ps = torch.exp(outputs)
        probs, indices = ps.topk(topk)
        probs = probs.squeeze()
        classes = [model.idx_to_class[idx] for idx in indices[0].tolist()]
    
    return probs, classes

#new_image_path = 'flowers/test/1/image_06743.jpg'
path = test_dir + '/1/image_06743.jpg'
predict(path, model)