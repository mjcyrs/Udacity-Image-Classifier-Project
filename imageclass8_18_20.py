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
dataloaders = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle=True)
detailer = iter(dataloaders)



# build and train your network here
import torchvision.models as models
images, labels = next(iter(dataloaders))
model = models.vgg16(pretrained=True)
# freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = True# build and train your network here
import torchvision.models as models
images, labels = next(iter(dataloaders))
model = models.vgg16(pretrained=True)
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
torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to = device
# define loss
criterion = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    print('cuda is available')
# removes randperm error from training pass  
cpu = torch.device('cpu')
tensor = torch.randperm(1, device=cpu)
new_tensor = tensor.cuda()


# Train the classifier layers using backpropagation using the pre-trained network to get the features
optimizer = optim.SGD(model.parameters(), lr=0.01)
#training loop
for epoch in range(2):
    running_loss = 0.0
    for images, labels in detailer:
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        print(running_loss)
print(running_loss)       
print('Finished Training')

# TODO: Do validation on the test set
test_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])
testset = datasets.ImageFolder(test_dir, transform = test_transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
# get test data
images = images.to('cuda')
labels = labels.to('cuda')
images, labels = next(iter(testloader))
# get class probabilities
ps = torch.exp(model(images))
print(ps.shape)
# running above code generates the error described