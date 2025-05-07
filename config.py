import argparse

def get_config():
    parser = argparse.ArgumentParser(description='CIFAR-10 Classification')
    
    # Training parameters
    parser.add_argument('--model', type=str, default='vgg', choices=['alexnet', 'densenet', 'vgg', 'resnet', 'mobilenet'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # Data parameters
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    
    # Path parameters
    parser.add_argument('--save_dir', type=str, default='saved_models')
    
    return parser.parse_args()