import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.config import config
import os
import sys


def get_cifar10_data():
    data_conf = config.get('data')
    transforms_conf = data_conf.get('transforms')

    transform = transforms.Compose([
        transforms.Resize(transforms_conf.get('resize')),  # Resize to 32x32 (CIFAR-10 original size)
        transforms.CenterCrop(transforms_conf.get('center_crop')),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=transforms_conf.get('normalize', {}).get('mean'),
            std=transforms_conf.get('normalize', {}).get('std')
        ),
    ])

    data_root = data_conf.get('data_root')
    train_dir = os.path.join(data_root, 'train')
    test_dir = os.path.join(data_root, 'test')

    # Check if dataset exists
    if not os.path.exists(os.path.join(data_root, 'cifar-10-batches-py')):
        print(f"CIFAR-10 dataset not found in {data_root}. Downloading automatically...")
        try:
            train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        except Exception as e:
            print(f"Failed to download CIFAR-10 dataset: {e}")
            sys.exit(1)
    else:
        # Load existing dataset
        train_dataset = datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform)
        test_dataset = datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_conf.get('batch_size'),
        shuffle=True,
        num_workers=data_conf.get('num_workers')
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_conf.get('batch_size'),
        shuffle=False,
        num_workers=data_conf.get('num_workers')
    )

    return train_loader, test_loader
#
# def get_imagenet_data():
#     data_conf = config.get('data')
#     transforms_conf = data_conf.get('transforms')
#
#     transform = transforms.Compose([
#         transforms.Resize(transforms_conf.get('resize')),
#         transforms.CenterCrop(transforms_conf.get('center_crop')),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=transforms_conf.get('normalize', {}).get('mean'),
#             std=transforms_conf.get('normalize', {}).get('std')
#         ),
#     ])
#
#     # Ensure ImageNet data exists
#     train_dir = os.path.join(data_conf.get('data_root'), 'train')
#     val_dir = os.path.join(data_conf.get('data_root'), 'val')
#     if not os.path.exists(train_dir) or not os.path.exists(val_dir):
#         print(f"ImageNet dataset not found in {data_conf.get('data_root')}. Please download it manually.")
#         sys.exit(1)
#
#     train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
#     test_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
#
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=data_conf.get('batch_size'),
#         shuffle=True,
#         num_workers=data_conf.get('num_workers')
#     )
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=data_conf.get('batch_size'),
#         shuffle=False,
#         num_workers=data_conf.get('num_workers')
#     )
#
#     return train_loader, test_loader
