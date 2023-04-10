import torch
from torchvision import transforms, datasets
import pathlib
from constants import batch_size


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        image_name = pathlib.Path(path).stem
        tuple_with_image_name = (original_tuple + (image_name,))
        return tuple_with_image_name


# define the standard image transforms
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# training and validation datasets and dataloaders
train_dataset = ImageFolderWithPaths(
    root='C:\\Users\\AntonioM\\Desktop\\dataset\\train',
    transform=train_transform
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_dataset = ImageFolderWithPaths(
    #root='C:\\Users\\AntonioM\\Desktop\\dataset\\val',
    root='C:\\Users\\AntonioM\\Desktop\\dataset_test\\',
    transform=val_transform
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)

print('Classes: ', train_dataset.classes)

"""
Dataset format


for i, data in enumerate(train_dataloader):
  images, labels, images_names = data
"""
