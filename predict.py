import sys
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
import numpy as np
from get_input_args import get_input_args

### Load the checkpoint 
def load_checkpoint(cp_dir, learn_rate):
    checkpoint = torch.load(cp_dir)
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    learning_rate = checkpoint['learning_rate']
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    optimizer.load_state_dict(checkpoint['optimizer_dict'])

    return model

### Use gpu or cpu according to user argument
def device_sel(gpu):
    device = torch.device("cuda:0" if gpu else "cpu")
    
    return device

### Maps image library as json
def load_mapping(category_name):
    with open(category_name, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

### Processes image for use in class prediction
def process_image(new_image):
    image = Image.open(new_image)
    width, height = image.size
    aspect_ratio = width / height
    if width > height:
        new_height = 256
        new_width = int(aspect_ratio * new_height)
    elif width < height:
        new_width = 256
        new_height = int(new_width / aspect_ratio)
    else:
        new_width = new_height = 256
    size = (new_width, new_height)
    image.thumbnail(size, Image.ANTIALIAS)
    image = image.crop((
        size[0]//2 - 112,
        size[1]//2 - 112,
        size[0]//2 + 112,
        size[1]//2 + 112)
    )
    np_image = np.array(image)
    #Scale Image per channel using (image-min)/(max-min)
    np_image = np_image/255.

    img_a = np_image[:,:,0]
    img_b = np_image[:,:,1]
    img_c = np_image[:,:,2]

    # Normalize image per channel
    img_a = (img_a - 0.485)/(0.229) 
    img_b = (img_b - 0.456)/(0.224)
    img_c = (img_c - 0.406)/(0.225)

    np_image[:,:,0] = img_a
    np_image[:,:,1] = img_b
    np_image[:,:,2] = img_c
    
    # Transpose image
    np_image = np.transpose(np_image, (2,0,1))
    return np_image

### Function to properly show the image
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = process_image(image)
    image = image.transpose((1, 2, 0))    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)    
    ax.imshow(image)    
    return ax

### Predict the class (or classes) of an image using a trained deep learning model
def predict(model, image_path, topk, cat_to_name):    
    model.to('cpu')
    model.eval()
    with torch.no_grad():
        image = image_path
        image = torch.from_numpy(np.array([image]))        
        image = image.float()
        logps = model(image)
        ps = torch.exp(logps)
        p, classes = ps.topk(topk, dim=1)
        top_p = p.tolist()[0]
        top_classes = classes.tolist()[0]
        idx_to_class = {v:k for k, v in model.class_to_idx.items()}
        labels = []
        for c in top_classes:
            labels.append(cat_to_name[idx_to_class[c]])

        return top_p, labels
    
def main():
    cmd_arg = get_input_args()
    
    model = load_checkpoint(cmd_arg.cp_dir, cmd_arg.learn_rate)
    device = device_sel(cmd_arg.predict_gpu)
    cat_to_name = load_mapping('cat_to_name.json')
    np_image = process_image(cmd_arg.img_path)
    top_p, labels = predict(model, np_image, cmd_arg.top_classes, cat_to_name)
    print(top_p, labels)
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)