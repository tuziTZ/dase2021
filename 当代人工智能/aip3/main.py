import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from lenet import LeNet
from alexnet import AlexNet
from resnet import ResNet
from vgg import VGG
from wideresnet import WideResNet
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_and_save_model(trained_model, train_data_loader, val_data_loader, loss_criterion, optimizer, num_epochs,
                         model_save_path, pic_save_path, args):
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        trained_model.train()
        # Iterate over the training dataset
        with tqdm(train_data_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as t:
            for images, labels in t:
                images, labels = images.to(device), labels.to(device)  # 解决gpu问题

                optimizer.zero_grad()
                outputs = trained_model(images)
                loss = loss_criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                t.set_postfix({'Train Loss': loss.item()})

            num_params = count_parameters(trained_model)
            print(f"Epoch {epoch + 1}, Number of parameters in the model: {num_params}")

        # Calculate training loss
        trained_model.eval()
        train_loss = 0.0
        with torch.no_grad():
            for images, labels in train_data_loader:
                images, labels = images.to(device), labels.to(device)  # 解决gpu问题

                outputs = trained_model(images)
                loss = loss_criterion(outputs, labels)
                train_loss += loss.item()
        train_loss /= len(train_data_loader)
        train_losses.append(train_loss)

        # Calculate validation loss and accuracy
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_data_loader:
                images, labels = images.to(device), labels.to(device)  # 解决gpu问题

                outputs = trained_model(images)
                loss = loss_criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_data_loader)
        val_losses.append(val_loss)

        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)

        print()
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    # Plotting the training and validation loss
    plot_loss_curve(train_losses, val_losses, pic_save_path, args)

    # Save the trained model
    save_model(trained_model, model_save_path, args)


def evaluate_model(trained_model, data_loader, dataset_name):
    trained_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)  # 解决gpu问题

            outputs = trained_model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'{dataset_name} Accuracy: {accuracy:.4f}')


def plot_loss_curve(train_losses, val_losses, pic_save_path, args):
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(pic_save_path,
                             f'{args.model}_train_val_loss_lr{args.lr}_dropout{args.dropout}_epoch{args.epochs}.png'))
    plt.close()


def save_model(trained_model, model_save_path, args):
    model_filename = f'{args.model}_lr{args.lr}_dropout{args.dropout}_epoch{args.epochs}.pth'
    torch.save(trained_model.state_dict(), os.path.join(model_save_path, model_filename))
    print(f'Model saved as {model_filename}')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description='Train a neural network on MNIST.')
    parser.add_argument('--model', type=str, default='vggnet', help='Model type')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=8, help='Number of epochs')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    args = parser.parse_args()
    torch.cuda.empty_cache()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)

    train_size = int(0.8 * len(train_dataset))  # 划分训练测试集
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    trained_model = ''
    if args.model == 'lenet':
        trained_model = LeNet(dropout_rate=args.dropout).to(device)
    elif args.model == 'alexnet':
        trained_model = AlexNet(dropout_rate=args.dropout).to(device)
    elif args.model == 'resnet':
        trained_model = ResNet().to(device)
    elif args.model == 'vggnet':
        trained_model = VGG().to(device)
    elif args.model == 'wideresnet':
        trained_model = WideResNet(dropout_rate=args.dropout).to(device)

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trained_model.parameters(), lr=args.lr)

    model_save_path = './model'
    pic_save_path = './pic'
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(pic_save_path, exist_ok=True)

    ts = time.time()
    train_and_save_model(trained_model, train_loader, val_loader, loss_criterion, optimizer, args.epochs,
                         model_save_path, pic_save_path, args)
    te = time.time()
    training_time = te - ts  # 计算训练用时
    print(f"Training completed in {training_time:.2f} seconds.")
    evaluate_model(trained_model, train_loader, 'Train Set')
    evaluate_model(trained_model, val_loader, 'Validation Set')
    evaluate_model(trained_model, test_loader, 'Test Set')


main()
