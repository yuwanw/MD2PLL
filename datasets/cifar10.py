import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np
import random

# 定义全局数据集
def Addnoise(labels, noise_rate, fake_number):

    K = int(torch.max(labels) - torch.min(labels) + 1)
    n = labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), labels] = 1.0
    new_y = partialY
    l, c = new_y.shape[0], new_y.shape[1]
    for i in range(l):
        ran = random.random()
        if ran < noise_rate:
            row = partialY[i, :]
            zero_index = np.where(row == 0)
            if len(zero_index[0]) >= fake_number:
                np.random.shuffle(zero_index[0])
                row[zero_index[0][:fake_number]] = 1
                new_y[i] = row

    return new_y
class GlobalDataset:
    def __init__(self, initial_data, initial_partialY, initial_true_labels):
        self.data = list(initial_data)  # 存储所有数据
        self.partialY = list(initial_partialY)  # 存储所有部分标签
        self.true_labels = list(initial_true_labels)  # 存储所有真实标签

    def add_data(self, new_data, new_partialY, new_true_labels):
        self.data.extend(new_data)  # 动态添加新数据
        self.partialY.extend(new_partialY)  # 动态添加新部分标签
        self.true_labels.extend(new_true_labels)  # 动态添加新真实标签

    def get_all_data(self):
        return self.data, self.partialY, self.true_labels  # 返回所有数据、部分标签和真实标签

# 定义动态验证集
class ValidAugmentation(data.Dataset):
    def __init__(self, data, partialY, true_labels):
        self.data = data  # 数据
        self.partialY = partialY  # 部分标签
        self.true_labels = true_labels  # 真实标签
        self.weak_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        image = self.weak_transform(self.data[index])
        partial_label = self.partialY[index]
        true_label = self.true_labels[index]
        return image, partial_label, true_label, index

# 定义静态训练集
class MNISTAugmentation(data.Dataset):
    def __init__(self, data, partialY, true_labels):
        self.data = data  # 数据
        self.partialY = partialY  # 部分标签
        self.true_labels = true_labels  # 真实标签
        self.weak_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ]
        )
        self.strong_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ]
        )

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        image_w = self.weak_transform(self.data[index])
        image_s = self.strong_transform(self.data[index])
        partial_label = self.partialY[index]
        true_label = self.true_labels[index]
        return image_w, image_s, partial_label, true_label, index

# 定义静态测试集
class TestAugmentation(data.Dataset):
    def __init__(self, data, true_labels):
        self.data = data  # 数据
        self.true_labels = true_labels  # 真实标签
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ]
        )

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        image = self.transform(self.data[index])
        true_label = self.true_labels[index]
        return image, true_label

# 动态挑选验证集数据的函数
def select_validation_data(global_dataset):
    data, partialY, true_labels = global_dataset.get_all_data()
    partialY = np.array(partialY)
    D = partialY.sum(1)  # 计算每个样本的候选标签数量
    index1 = np.where(D == 1)[0]  # 挑选候选标签数量为 1 的样本
    v_data = [data[i] for i in index1]
    v_partialY = [partialY[i] for i in index1]
    v_true_labels = [true_labels[i] for i in index1]
    return v_data, v_partialY, v_true_labels

# 加载 CIFAR-10 数据集
def load_cifar10(batch_size, partial_rate):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    train_dataset = dsets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    test_dataset = dsets.CIFAR10(root='./data', train=False, transform=transform)

    train_data = train_dataset.data
    train_labels = torch.tensor(train_dataset.targets).long()

    # 生成部分标签
    partialY = Addnoise(labels=train_labels, noise_rate=partial_rate, fake_number=2)

    # 检查部分标签是否正确生成
    temp = torch.zeros(partialY.shape)
    temp[torch.arange(partialY.shape[0]), train_labels] = 1
    # if torch.sum(partialY * temp) == partialY.shape[0]:
    #     print('Partial labels correctly loaded')
    # else:
    #     print('Inconsistent permutation')
    #
    # print('Average candidate num: ', partialY.sum(1).mean())

    # 创建全局数据集
    global_dataset = GlobalDataset(train_data, partialY, train_labels)

    # 创建静态训练集
    train_dataset = MNISTAugmentation(train_data, partialY, train_labels)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 创建静态测试集
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size * 4, shuffle=False, num_workers=4)

    return global_dataset, train_loader, test_loader