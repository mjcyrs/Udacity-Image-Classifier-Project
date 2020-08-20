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
optimizer = optim.SGD(model.parameters(), lr=0.001)
epochs = 10
#training loop
for epoch in range(epochs):
    running_loss = 0.0
    accuracy = 0
    for images, labels in detailer:
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

# TODO: Do validation on the test set
test_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])
testset = datasets.ImageFolder(valid_dir, transform = test_transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
model.eval()
model.cuda()
images = images.to('cuda')
labels = labels.to('cuda')
test_loss = 0
test_accuracy = 0
# test loop
with torch.no_grad():
    for images, labels in testloader:
        log_ps = model(images)
        test_loss +- criterion(log_ps, labels)
        
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(i, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        test_accuracy +=torch.mean(equals.type(torch.FloatTensor))        
print("Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
"Test Accuracy: {:.3f}".format(test_accuracy/len(testloader)))