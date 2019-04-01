import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


def mnist(batch_size=100, download=True):
    data_transforms = [transforms.ToTensor()]

    train_transforms = data_transforms
    test_transforms = data_transforms
    train_transform = transforms.Compose(train_transforms)
    test_transform = transforms.Compose(test_transforms)

    train_set = MNIST(root='./data', train=True, download=download,
                      transform=train_transform)

    total_train_samples = len(train_set)
    total_val_samples = 5000

    val_set = MNIST(root='./data', train=True, download=download,
                    transform=test_transform)
    test_set = MNIST(root='./data', train=False, download=download,
                     transform=test_transform)

    train_sampler = SubsetRandomSampler(
        list(range(total_train_samples - total_val_samples)))
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=False, num_workers=10,
                                              sampler=train_sampler)

    total_train_samples = len(val_set)
    val_sampler = SubsetRandomSampler(
        list(range(total_train_samples - total_val_samples,
                   total_train_samples)))
    valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                            shuffle=False, num_workers=10,
                                            sampler=val_sampler)

    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                             shuffle=False, num_workers=10)

    return trainloader, valloader, testloader
