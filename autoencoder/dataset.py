import torch
from torchvision import transforms, datasets


images_size = 224

# define the standard image transforms
train_transform = transforms.Compose([
        transforms.Resize(images_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])

val_transform = transforms.Compose([
        transforms.Resize(images_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# training and validation datasets and dataloaders
train_dataset = datasets.ImageFolder(
    root='C:\\Users\\AntonioM\\Desktop\\dataset\\train',
    transform=train_transform
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, shuffle=True
)
val_dataset = datasets.ImageFolder(
    root='C:\\Users\\AntonioM\\Desktop\\dataset\\val',
    transform=val_transform
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=16, shuffle=False
)

# print('Classes: ', train_dataset.classes)
