import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import os
import torch
from torchvision import transforms
from lenet import LeNet
from alexnet import AlexNet
from resnet import ResNet
from vgg import VGG
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import struct

def load_model(model_path, model_type, dropout_rate):
    if model_type == 'lenet':
        model = LeNet(dropout_rate=dropout_rate)
    elif model_type == 'alexnet':
        model = AlexNet(dropout_rate=dropout_rate)
    elif model_type == 'resnet':
        model = ResNet()
    elif model_type == 'vgg':
        model = VGG()
    elif model_type == 'googlelenet':
        model = GoogleLeNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def read_idx3_ubyte(file_path):
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    return images

def predict_image(model, image, transform):
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0)  # 添加批处理维度
    output = model(image)
    _, predicted_class = torch.max(output, 1)
    confidence = torch.nn.functional.softmax(output, dim=1)[0] * 100
    return predicted_class.item(), confidence[predicted_class].item()

def visualize_prediction(image, class_name, confidence):
    plt.imshow(image, cmap='gray')
    plt.title(f'class: {class_name}, confidence: {confidence:.2f}%')
    plt.axis('off')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='使用训练好的模型在MNIST测试数据上进行对象检测.')
    parser.add_argument('--model-path', type=str, default='model/alexnet_lr0.001_dropout0.5_epoch3.pth', help='训练好的模型文件路径')
    parser.add_argument('--model-type', type=str, default='alexnet', help='模型类型（lenet或其他）')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout率')
    parser.add_argument('--data-file', type=str, default='data/MNIST/raw/t10k-images-idx3-ubyte', help='MNIST测试数据文件路径（t10k-images-idx3-ubyte）')
    parser.add_argument('--num-samples', type=int, default=3, help='要测试的随机样本数')
    parser.add_argument('--seed', type=int, default=42, help='随机数种子以确保可重复性')

    args = parser.parse_args()

    # 设置随机数种子以确保可重复性
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 加载训练好的模型
    model = load_model(args.model_path, args.model_type, args.dropout)

    # 读取MNIST测试数据
    test_images = read_idx3_ubyte(args.data_file)

    # 定义数据转换
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # 从测试数据中随机选择样本
    selected_indices = random.sample(range(len(test_images)), min(args.num_samples, len(test_images)))

    # 对每个选择的样本进行预测和可视化
    for index in selected_indices:
        image = test_images[index]
        predicted_class, confidence = predict_image(model, image, transform)
        visualize_prediction(image, predicted_class, confidence)

if __name__ == "__main__":
    main()
