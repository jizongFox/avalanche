import os

import numpy as np
from torchvision import transforms

from haxio.data import ImageFolder
from haxio.data.augment import DownSample, RandomShift


def HaxioDataset(dataroot, train_aug=False):
    # sorted_classes = ["Good", "Damage", "Cap", "Hair", "Black"]
    sorted_classes = ["Good", "Empty", "Hair", "NoCap", "Particle", "Unfilled"]

    val_transform = transforms.Compose([
        DownSample(5),
        transforms.ToTensor(),
    ])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            DownSample(5),
            RandomShift(0.1, 0.1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    train_dataset = ImageFolder(
        root=os.path.join(dataroot, "train"),
        transform=train_transform, customized_classes=sorted_classes
    )
    val_dataset = ImageFolder(
        root=os.path.join(dataroot, "val"), customized_classes=sorted_classes,
        transform=val_transform
    )
    assert train_dataset.classes == val_dataset.classes

    return train_dataset, val_dataset


def dataset_descript(experience):
    dataset = experience.dataset
    target_classes = np.unique([x[1] for x in dataset])
    task_classes = np.unique([x[2] for x in dataset])
    print("target_classes:", target_classes, "task_classes", task_classes)
