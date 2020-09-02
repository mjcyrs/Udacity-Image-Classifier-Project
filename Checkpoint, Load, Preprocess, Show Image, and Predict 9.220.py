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
    
    model.idx_to_class = dict(map(reversed, model.class_to_idx.items()))
    
    with torch.no_grad():
        outputs = model.forward(image)
        ps = torch.exp(outputs)
        probs, indices = ps.topk(topk)
        probs = probs.squeeze()
        classes = [model.idx_to_class[idx] for idx in indices[0].tolist()]
    
    return probs, classes

image_path = process_image('flowers/test/1/image_06743.jpg')
predict(image_path, classifier)