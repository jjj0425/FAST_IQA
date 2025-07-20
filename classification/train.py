import os
import argparse
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import get_model
from tqdm import tqdm

def train_model(
    model: torch.nn.Module,
    dataloaders: dict,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    log_file: str
):
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    logging.basicConfig(filename=log_file, level=logging.INFO)
    logging.info("Starting training...")

    for epoch in range(num_epochs):
        logging.info(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            with tqdm(total=len(dataloaders[phase]), desc=f'{phase.capitalize()} Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    pbar.update(1)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

            logging.info(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

def main():
    parser = argparse.ArgumentParser(description="Train a model on a dataset")
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--model', type=str, required=True, choices=['vgg16', 'resnet50', 'mobilenetv2', 'googlenet'], help='Model name')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'training.log')

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=20),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(args.data_root, x), data_transforms[x])
        for x in ['train', 'val']
    }
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=8),
        'val': DataLoader(image_datasets['val'], batch_size=args.batch_size, shuffle=False, num_workers=8)
    }

    print(f'Number of training images: {len(image_datasets["train"])}')
    print(f'Number of validation images: {len(image_datasets["val"])}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model, num_classes=5).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    model, train_loss_history, val_loss_history, train_acc_history, val_acc_history = train_model(
        model, dataloaders, criterion, optimizer, args.num_epochs, device, log_file)

    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pth'))

    plt.figure()
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'loss_curve.png'))

    plt.figure()
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'accuracy_curve.png'))

if __name__ == "__main__":
    main()
