import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import os

def get_dataloaders(config):
    # Data augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Custom data paths
    train_data_path = "./data/train"
    test_data_path = "./data/test"

    # Load dataset từ thư mục
    full_dataset = datasets.ImageFolder(
        root=train_data_path,
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        root=test_data_path,
        transform=test_transform
    )

    # Split train/valid
    train_size = int(config.train_ratio * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=4
    )

    return train_loader, valid_loader, test_loader