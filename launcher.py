import sys
from train import load_mapping, image_load, architecture, train_eval, test_loop, save_cpt
from predict import load_checkpoint, process_image, imshow, predict
from get_input_args import get_input_args

def main():
    cmd_arg = get_input_args()  
    load_mapping('cat_to_name.json')
    image_load()
    architecture(cmd_arg.cnn, cmd_arg.hidden_layer, cmd_arg.gpu, cmd_arg.learn_rate)
    train_eval(cmd_arg.number_epochs)
    test_loop()
    save_cpt(cmd_arg.cnn, cmd_arg.hidden_layer, cmd_arg.learn_rate, cmd_arg.number_epochs, cmd_arg.cp_dir)
    
    process_image('flowers/test/60/image_02971.jpg')

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)