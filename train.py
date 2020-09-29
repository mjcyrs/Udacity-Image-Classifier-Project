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

### Maps image library as json
def load_mapping(category_name):
    with open(category_name, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name

### Loads and transforms images
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

    return image_datasets, dataloaders

### Defines the bulk architecture of the model
def architecture(cnn, hidden_layer, gpu, learn_rate):
    choice = cnn
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
    # define classifier layers and details
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                               ('fc1', nn.Linear(25088, hidden_layer)),
                               ('relu', nn.ReLU()),
                               ('dropout1', nn.Dropout(p=0.2)),
                               ('fc2', nn.Linear(hidden_layer, 102)),
                               ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    
    return model, criterion, optimizer

### Use gpu or cpu according to user argument
def device_sel(gpu):
    device = torch.device("cuda:0" if gpu else "cpu")
    
    return device
    
### Train and validate the network
def train_eval(device, model, number_epochs, optimizer, criterion, learn_rate, dataloaders, validloader):   
    model.to(device)
    model.train()
    epochs = number_epochs
    steps = 0
    print_every = 60
    running_loss = 0
    #training loop
    for epoch in range(epochs):
        for images, labels in dataloaders:
            running_loss = 0
            steps += 1
            images = images.to(device)
            labels = labels.to(device)
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
                        ps = torch.exp(validps)
                        top_prob, top_label = ps.topk(1, dim=1)
                        equals = top_label == labels.view(*top_label.shape)
                        accuracy += torch.mean(equals.type(torch.cuda.FloatTensor)).item()                        
                    print("Epoch: {}/{}:".format(epoch + 1, epochs),
                            f"Training_loss: {running_loss/print_every} "
                            f"Validation loss: {valid_loss/len(validloader):.3f} "
                            f"Validation Accuracy: {accuracy/len(validloader):.3f}")
            model.train()        
    print('finished training')

### Test network
def test_loop(model, device, testloader):
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

### Save the checkpoint and rebuild the model
def save_cpt(model, image_datasets, gpu, hidden_layer, learn_rate, number_epochs, optimizer, cp_dir):
    model.class_to_idx = image_datasets.class_to_idx
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
                  'class_to_idx': image_datasets.class_to_idx,
                  'optimizer_dict': optimizer.state_dict()}
    torch.save(checkpoint, cp_dir)
    
    return checkpoint

def main():
    cmd_arg = get_input_args()
    
    load_mapping('cat_to_name.json')
    image_datasets, dataloaders = image_load()
    device = device_sel(cmd_arg.gpu)
    model, criterion, optimizer = architecture(cmd_arg.cnn, cmd_arg.hidden_layer, cmd_arg.gpu, cmd_arg.learn_rate)
    train_eval(device, model, cmd_arg.number_epochs, optimizer, criterion, cmd_arg.learn_rate, dataloaders['train'], dataloaders['valid'])
    test_loop(model, device, dataloaders['test'])
    save_cpt(model, image_datasets['test'], cmd_arg.cnn, cmd_arg.hidden_layer, cmd_arg.learn_rate, cmd_arg.number_epochs, optimizer, cmd_arg.cp_dir)
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)