import numpy as np
import torch
import torchvision.datasets as datasets

class ImageFolderInstance(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super(ImageFolderInstance, self).__init__(root, transform, target_transform)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        image = self.loader(path)
        if self.transform is not None:
            img = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index



class ImageFolderInstance_LoadAllImgToMemory(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super(ImageFolderInstance, self).__init__(root, transform, target_transform)
        self.allImgs = []
        lastP = -1
        for index in range(len(self.imgs)):
            path, target = self.imgs[index]
            image = self.loader(path)
            if self.transform is not None:
                img = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)
                
            self.allImgs.append(img)
            percentage = int((float(index) / len(self.imgs))*100)
            if percentage%10==0 and lastP != percentage:
                lastP = percentage
                print('Loading data %d%%'%percentage)
        print('Data loading finished.')


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        # image = self.loader(path)
        # if self.transform is not None:
        #     img = self.transform(image)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        img = self.allImgs[index]

        return img, target, index