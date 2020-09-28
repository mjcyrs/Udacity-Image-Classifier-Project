import argparse

def get_input_args():
    parser = argparse.ArgumentParser()
    
    #Train Arguments
    parser.add_argument('--cnn', type = str, default = 'vgg16', dest = 'cnn', help = 'CNN model architecture (choose resnet, alexnet, or vgg)')
    parser.add_argument('--lr', type = float, default = 0.001, dest = 'learn_rate', help = 'Learning Rate')
    parser.add_argument('--hlu', type = int, default = 4096, dest = 'hidden_layer', help = 'Number of nodes in the hidden layer')
    parser.add_argument('--epochs', type = int, default = 8, dest = 'number_epochs', help = 'Number of Epochs in Training Loop')
    parser.add_argument('--gpu', type = bool, default = True, dest = 'gpu', help = 'Use GPU (True) or CPU (False) on train data')
    
    #Predict Arguments
    parser.add_argument('--cpt', type = str, default = 'bigcatmodel.pth', dest = 'cp_dir', help = 'load a checkpoint')
    parser.add_argument('--top_k', type = int, default = 5, dest = 'top_classes', help = 'List of most likely classifications')
    parser.add_argument('--cat_names', type = str, default = 'cat_to_name.json', dest = 'category_name', help = 'Categories to flower names - json file')
    parser.add_argument('--pred_gpu', type = bool, default = True, dest = 'predict_gpu', help = 'Use GPU (True) or CPU (False) for prediction')
    
    return parser.parse_args()