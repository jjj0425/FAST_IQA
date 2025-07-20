import os
import random
from typing import Tuple, List, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.images: List[str] = []
        self.labels: List[int] = []
        self.filenames: List[str] = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.images.append(img_path)
                self.labels.append(label)
                self.filenames.append(img_name)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def save_split_info(root_dir: str, train_ratio: float) -> None:
    dataset = CustomDataset(root_dir=root_dir)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split_idx = int(train_ratio * dataset_size)
    random.shuffle(indices)

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_filenames = [dataset.filenames[i] for i in train_indices]
    val_filenames = [dataset.filenames[i] for i in val_indices]

    save_dir = 'run'
    os.makedirs(save_dir, exist_ok=True)

    # Save split information to a file
    with open(os.path.join(save_dir, 'split_info.txt'), 'w') as f:
        f.write("Training images:\n")
        f.write('\n'.join(train_filenames) + '\n\n')
        f.write("Validation images:\n")
        f.write('\n'.join(val_filenames) + '\n\n')

def get_dataloaders(
    root_dir: str,
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
    batch_size: int = 32,
    train_ratio: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    dataset = CustomDataset(root_dir=root_dir, transform=train_transform)
    save_split_info(root_dir, train_ratio)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split_idx = int(train_ratio * dataset_size)

    train_sampler = SubsetRandomSampler(indices[:split_idx])
    val_sampler = SubsetRandomSampler(indices[split_idx:])

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    return train_loader, val_loader

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
