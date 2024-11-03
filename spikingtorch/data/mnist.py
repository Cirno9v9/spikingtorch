import torch
import torchvision
from torch.utils import data
import os

def load_mnist(data_dir: str, b: int = 1, j: int = 4):
    """
    加载mnist数据集，返回train和test的loader
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f'路径 "{os.path.abspath(data_dir)}" 不存在')
    
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=b,
        shuffle=True,
        drop_last=True,
        num_workers=j,
        pin_memory=True
    )
    test_data_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=b,
        shuffle=False,
        drop_last=False,
        num_workers=j,
        pin_memory=True
    )
    return train_data_loader, test_data_loader