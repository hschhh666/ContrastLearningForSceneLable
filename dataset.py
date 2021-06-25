import numpy as np
import torch
import torchvision.datasets as datasets

class ImageFolderInstance(datasets.ImageFolder):
    def __init__(self, args, root, transform=None, target_transform=None):
        super(ImageFolderInstance, self).__init__(root, transform, target_transform)
        self.flow_dict = np.load(args.optical_flow_npy_path, allow_pickle=True).item()


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

        # 是灰度图，仅读取一个通道就行了
        img = img[0,:,:].unsqueeze(0)
        # 读取并归一化光流
        img_name = path.split('/')[-1]
        flow = self.flow_dict[img_name]
        # 计算光流的模，即忽略光流方向，仅看光流强度
        flow_mod = np.sqrt(np.power(flow[:,:,0], 2) + np.power(flow[:,:,1], 2))
        # 归一化，均值和方差是通过离线计算得到的
        flow_mod = (flow_mod - 2.451753) / 7.033634
        flow_mod = flow_mod[np.newaxis,:]
        flow_mod = torch.tensor(flow_mod)
        img = torch.cat((img, flow_mod))

        return img, target, index

class ImageFolderInstance_LoadAllImgToMemory(datasets.ImageFolder):
    def __init__(self, args, root, transform=None, target_transform=None, training_data_cache_method = 'memory'):
        super(ImageFolderInstance_LoadAllImgToMemory, self).__init__(root, transform, target_transform)
        self.allImg = []
        self.flow_dict = np.load(args.optical_flow_npy_path, allow_pickle=True).item()
        lastP = -1
        self.training_data_cache_method = training_data_cache_method
        for index in range(len(self.imgs)):
            path, target = self.imgs[index]
            img = self.loader(path)
            if self.training_data_cache_method == 'GPU' and self.transform is not None:
                img = self.transform(img)
                img = img.cuda()
            if self.target_transform is not None:
                target = self.target_transform(target)
                
            self.allImg.append(img)
            percentage = int((float(index) / len(self.imgs))*100)
            if percentage % 10 == 0 and lastP != percentage:
                lastP = percentage
                print('Loading data %d%%'%percentage)
        print('Data loading finished.')

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.allImg[index]
        if self.training_data_cache_method == 'memory' and self.transform is not None:
            img = self.transform(img)

        # 是灰度图，仅读取一个通道就行了
        img = img[0,:,:].unsqueeze(0)
        # 读取并归一化光流
        img_name = path.split('/')[-1]
        flow = self.flow_dict[img_name]
        # 计算光流的模，即忽略光流方向，仅看光流强度
        flow_mod = np.sqrt(np.power(flow[:,:,0], 2) + np.power(flow[:,:,1], 2))
        # 归一化，均值和方差是通过离线计算得到的
        flow_mod = (flow_mod - 2.451753) / 7.033634
        flow_mod = flow_mod[np.newaxis,:]
        flow_mod = torch.tensor(flow_mod)
        img = torch.cat((img, flow_mod))

        return img, target, index