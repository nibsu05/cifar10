import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from utils.dataloader import get_dataloaders
from utils.metrics import calculate_metrics
from models import VGG, AlexNet, DenseNet, ResNet, MobileNet
from config import get_config

def test():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model_path = f"{config.save_dir}/best_{config.model}.pth"
    
    if config.model == 'vgg':
        model = VGG().to(device)
    elif config.model == 'alexnet':
        model = AlexNet().to(device)
    elif config.model == 'densenet':
        model = DenseNet().to(device)
    elif config.model == 'resnet':
        model = ResNet().to(device)
    elif config.model == 'mobilenet':
        model = MobileNet().to(device)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load data
    _, _, test_loader = get_dataloaders(config)
    
    # Testing
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    print(classification_report(all_labels, all_preds, target_names=test_loader.dataset.classes))
    calculate_metrics(all_labels, all_preds)

if __name__ == '__main__':
    test()