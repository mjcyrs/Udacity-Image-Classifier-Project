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

class predictor:
    def __init__(self):
        pass
       
    ### Load the checkpoint 
    def load_checkpoint(self, cp_dir):
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
        #use below to test, elsewise ignore
        #file_name = 'bigcatmodel.pth'
        #saved_model = load_checkpoint(file_name)

        ### Classification
        ### Preprocess Image for successful passing into the model
    def process_image(self, new_image):

        # TODO: Process a PIL image for use in a PyTorch model
        image = Image.open(new_image)
        size = 256, 256
        image.thumbnail(size, Image.ANTIALIAS)
        image = image.crop((
            size[0]//2 - 112,
            size[1]//2 - 112,
            size[0]//2 + 112,
            size[1]//2 + 112)
        )
        np_image = np.array(image)
        #Scale Image per channel
        # Using (image-min)/(max-min)
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
    def imshow(self, image, ax=None, title=None):
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

        ### Predict function for classification
    def predict(self, image_path, model, checkpoint, predict_gpu):
        ''' Predict the class (or classes) of an image using a trained deep learning model.'''
        # TODO: Implement the code to predict the class from an image file
        load_checkpoint(checkpoint)
        topk = top_classes
        device = torch.device("cuda:0" if predict_gpu else "cpu")
        model = model.to(device)
        model.eval()
        with torch.no_grad():

            image = process_image(image_path)
            image = torch.from_numpy(np.array([image]))

            #image = image.unsqueeze(0)
            image = image.float()
            #image = image.to('cpu')

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