import torch
from PIL import Image
from torchvision import transforms
import argparse
from models import VGG, AlexNet, DenseNet

def predict(image_path, model_name='vgg'):
    # Thiết lập device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Khởi tạo model
    if model_name == 'vgg':
        model = VGG().to(device)
    elif model_name == 'alexnet':
        model = AlexNet().to(device)
    elif model_name == 'densenet':
        model = DenseNet().to(device)
    else:
        raise ValueError(f"Model {model_name} không tồn tại")

    # Load weights
    model_path = f"saved_models/best_{model_name}.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Xử lý ảnh
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Dự đoán
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"\nKết quả dự đoán cho {image_path}:")
    print(f"Model: {model_name.upper()}")
    print(f"→ Class: {class_names[pred.item()]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dự đoán ảnh CIFAR-10')
    parser.add_argument('image_path', type=str, help='Đường dẫn đến ảnh cần dự đoán')
    parser.add_argument('--model', type=str, default='vgg', 
                        choices=['alexnet', 'densenet', 'vgg'],
                        help='Lựa chọn model (mặc định: vgg)')
    
    args = parser.parse_args()
    predict(args.image_path, args.model)