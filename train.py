import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os
import json
from PIL import Image
import torchvision.models as models
import sys
from get_input_args import get_input_args

cmd_arg = get_input_args()

def load_mapping(category_name):
    with open(category_name, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name

def image_load():
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    ### Define Transforms for training, validation, and testing
    image_path = 'flowers/train/1/image_06734.jpg'
    image = Image.open(image_path)
    data_transforms = {'train': transforms.Compose([transforms.RandomRotation(10),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),                                                                             
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])]),

                        'test': transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])]),

                        'valid': transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])}

    image_datasets = {'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
                'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])}

    dataloaders ={'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size = 64, shuffle=True),
                    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True),
                    'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle =True)}

    #images, labels = next(iter(dataloaders))
    #return image_datasets, dataloaders

def architecture(cnn, hidden_layer, gpu, learn_rate):
    choice = cmd_arg.cnn
    if choice == 'vgg16':
        models.vgg16(pretrained=True)
        model = models.vgg16(pretrained=True)
    elif choice == 'resnet':
        models.resnet18(pretrained=True)
        model = models.resnet18(pretrained=True)
    elif choice == 'alexnet':
        models.alexnet(pretrained=True)
        model = models.alexnet(pretrained=True)
    else:
        print('choose vgg16, resnet18, or alexnet for your cnn, silly')
        sys.exit(1)

    # freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                               ('fc1', nn.Linear(25088, hidden_layer)),
                               ('relu', nn.ReLU()),
                               ('dropout1', nn.Dropout(p=0.2)),
                               ('fc2', nn.Linear(hidden_layer, 102)),
                               ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    device = torch.device("cuda:0" if gpu else "cpu")
    model = model.to(device)
    # define loss
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

    ### Train and validate the network
def train_eval(number_epochs):   
    #model.cuda()
    epochs = number_epochs
    steps = 0
    print_every = 60
    running_loss = 0
    #training loop
    for epoch in range(epochs):
        for images, labels in image_load.dataloaders['train']:
            running_loss = 0
            steps += 1
            images = images.cuda()
            labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            logps = model.forward(images)
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
                        #model.train()
                    print("Epoch: {}/{}:".format(epoch + 1, epochs),
                            f"Training_loss: {running_loss/print_every} "
                            f"Validation loss: {valid_loss/len(validloader):.3f} "
                            f"Validation Accuracy: {accuracy/len(validloader):.3f}")
            model.train()        
    print('finished training')

    ### Test network
def test_loop():
    model.eval()
    test_var = 0
    accuracy = 0
    #testloader = image_load(testloader)
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

    ### Save the checkpoint and rebuild the model
def save_cpt(gpu, hidden_layer, learn_rate, number_epochs, cp_dir):
    model.class_to_idx = trainset.class_to_idx
    device = torch.device("cuda:0" if gpu else "cpu")
    model = model.to(device)
    class_to_idx = model.class_to_idx

    checkpoint = {'input_size': 25088,
                  'hidden_layer_size': hidden_layer,
                  'output_size': 102,
                  'classifier': model.classifier,
                  'learning_rate': learn_rate,
                  'training_epochs': number_epochs,
                  'state_dict': model.state_dict(),
                  'class_to_idx': trainset.class_to_idx,
                  'optimizer_dict': optimizer.state_dict()}
    torch.save(checkpoint, cp_dir)

    