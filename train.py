import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from utils.dataloader import get_dataloaders
from utils.metrics import calculate_metrics
from utils.visualize import plot_training_curves
from models import VGG, AlexNet, DenseNet, ResNet, MobileNet
from config import get_config

def train():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_loader, valid_loader, _ = get_dataloaders(config)
    
    # Initialize model
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

    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Training loop
    best_acc = 0.0
    history = {'train_loss': [], 'valid_loss': [], 'train_acc': [], 'valid_acc': []}
    
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss/len(train_loader)
        train_acc = correct/total
        
        # Validation
        valid_loss, valid_acc = validate(model, valid_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)
        
        # Save best model
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), f"{config.save_dir}/best_{config.model}.pth")
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{config.epochs}')
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Valid Loss: {valid_loss:.4f} Acc: {valid_acc:.4f}\n')
    
    # Plot training curves
    plot_training_curves(history)

def validate(model, valid_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss/len(valid_loader), correct/total

if __name__ == '__main__':
    train()