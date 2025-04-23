import os
import numpy as np
import pickle
from PIL import Image

def convert_cifar10_to_images(data_path, output_dir):
    # Tạo thư mục
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    with open(data_path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    
    # Tạo thư mục class
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    for class_name in classes:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
    
    # Convert và lưu ảnh
    for i, (image, label) in enumerate(zip(data[b'data'], data[b'labels'])):
        # Reshape từ 3072 pixel (3x32x32) thành ảnh 32x32x3
        image = image.reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(image)
        
        class_name = classes[label]
        img.save(os.path.join(output_dir, class_name, f'image_{i:05d}.png'))

# Convert tất cả các batch
for batch in [f'data_batch_{i}' for i in range(1,6)] + ['test_batch']:
    convert_cifar10_to_images(
        data_path=f"./data/cifar-10-batches-py/{batch}",
        output_dir="./data/train" if 'data_batch' in batch else "./data/test"
    )