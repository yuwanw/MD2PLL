import torch.utils.data as data
from scipy.io import loadmat
import torch
import numpy as np
from sklearn.model_selection import KFold
from utils.utils_algo import *


class KFoldDataLoader:
    def __init__(self, mat_path, n_splits=5):
        self.n_splits = n_splits
        self.data = loadmat(mat_path)
        self.features, self.targets, self.partial_targets = self.data['data'], self.data['target'], self.data[
            'partial_target']

        # 确保数据形状一致
        if self.features.shape[0] != self.targets.shape[0]:
            self.targets = self.targets.transpose()
            self.partial_targets = self.partial_targets.transpose()
        if type(self.targets) != np.ndarray:
            self.targets = self.targets.toarray()
            self.partial_targets = self.partial_targets.toarray()

        # 归一化
        self.features = (self.features - self.features.mean(axis=0, keepdims=True)) / self.features.std(axis=0,
                                                                                                        keepdims=True)

        # KFold 划分
        self.kfold = KFold(n_splits=n_splits, random_state=0, shuffle=True)
        self.train_test_idx = [(tr_idx, te_idx) for tr_idx, te_idx in self.kfold.split(self.features)]
        self.num_features, self.num_classes = self.features.shape[-1], self.targets.shape[-1]

    def k_cross_validation(self, k: int):
        assert k >= 0 and k < self.n_splits
        self.tr_idx, self.te_idx = self.train_test_idx[k]
        self.train_features, self.train_targets, self.train_labels = self.features[self.tr_idx], self.partial_targets[
            self.tr_idx], self.targets[self.tr_idx]
        self.test_features, self.test_targets, self.test_labels = self.features[self.te_idx], self.partial_targets[
            self.te_idx], self.targets[self.te_idx]

        def to_sum_one(x):
            return x / x.sum(axis=1, keepdims=True)

        def to_torch(x):
            return torch.from_numpy(x).to(torch.float32)

        self.train_final_labels, self.test_final_labels = map(to_sum_one, (self.train_targets, self.test_targets))
        self.train_features, self.train_targets, self.train_final_labels, self.train_labels, self.test_features, self.test_targets, self.test_final_labels, self.test_labels = map(
            to_torch, (
                self.train_features, self.train_targets, self.train_final_labels, self.train_labels, self.test_features,
                self.test_targets, self.test_final_labels, self.test_labels))

        return (self.train_features, self.train_targets, self.train_final_labels, self.train_labels), (
            self.test_features, self.test_targets, self.test_final_labels, self.test_labels)


class RealWorldData(data.Dataset):
    def __init__(self, k, train_or_not, k_fold_dataloader):
        self.k = k
        self.train = train_or_not
        self.train_dataset, self.test_dataset = k_fold_dataloader.k_cross_validation(self.k)
        self.train_features, self.train_targets, self.train_final_labels, self.train_labels = self.train_dataset
        self.test_features, self.test_targets, self.test_final_labels, self.test_labels = self.test_dataset

    def __getitem__(self, index):
        if self.train:
            feature, target, final, true = self.train_features[index], self.train_targets[index], \
            self.train_final_labels[index], self.train_labels[index]
        else:
            feature, target, final, true = self.test_features[index], self.test_targets[index], self.test_final_labels[
                index], self.test_labels[index]
        return feature, target, final, true, index

    def __len__(self):
        if self.train:
            return len(self.train_features)
        else:
            return len(self.test_features)

class GlobalDataset:
    def __init__(self, initial_data, initial_labels, initial_partial_labels):
        self.data = list(initial_data)  # 存储所有数据
        self.labels = list(initial_labels)  # 存储所有真实标签
        self.partial_labels = list(initial_partial_labels)  # 存储所有部分标签

    def add_data(self, new_data, new_labels, new_partial_labels):
        self.data.extend(new_data)  # 动态添加新数据
        self.labels.extend(new_labels)  # 动态添加新真实标签
        self.partial_labels.extend(new_partial_labels)  # 动态添加新部分标签

    def get_all_data(self):
        return self.data, self.labels, self.partial_labels

def extract_data(dataname):
    mat_path = "data/" + dataname + '.mat'
    k_fold = KFoldDataLoader(mat_path)
    for k in range(0, 5):
        train_dataset = RealWorldData(k, True, k_fold)
        test_dataset = RealWorldData(k, False, k_fold)
        train_X = train_dataset.train_features
        train_Y = train_dataset.train_labels
        train_p_Y = train_dataset.train_targets
        test_X = test_dataset.test_features
        test_Y = test_dataset.test_labels
        yield train_X, train_Y, train_p_Y, test_X, test_Y


def create_train_loader(train_X, train_Y, train_p_Y, batch_size=64):
    class dataset(data.Dataset):
        def __init__(self, train_X, train_Y, train_p_Y):
            self.train_X = train_X
            self.train_p_Y = train_p_Y
            self.train_Y = train_Y

        def __len__(self):
            return len(self.train_X)

        def __getitem__(self, idx):
            if isinstance(self.train_X[idx], np.ndarray):
                self.train_X[idx] = torch.from_numpy(self.train_X[idx])  # 假设特征数据是浮点数
            if isinstance(self.train_p_Y[idx], np.ndarray):
                self.train_p_Y[idx] = torch.from_numpy(self.train_p_Y[idx]) # 假设标签数据是长整型（如果是分类问题）
            if isinstance(self.train_Y[idx], np.ndarray):
                self.train_Y[idx] = torch.from_numpy(self.train_Y[idx])
            return self.train_X[idx], self.train_p_Y[idx], self.train_Y[idx], idx

    ds = dataset(train_X, train_Y, train_p_Y)
    dl = data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    return dl


def select_validation_data(global_dataset):
    data, labels, partial_labels = global_dataset.get_all_data()
    data = torch.stack(data) if isinstance(data[0], torch.Tensor) else torch.tensor(data)
    labels = torch.stack(labels) if isinstance(labels[0], torch.Tensor) else torch.tensor(labels)
    partial_labels = torch.stack(partial_labels) if isinstance(partial_labels[0], torch.Tensor) else torch.tensor(partial_labels)
    # partial_labels = np.stack(partial_labels,axis=0)
    # 选择部分标签数量为 1 的样本作为验证集
    D = partial_labels.sum(1)
    index1 = torch.where(D == 1)[0]
    v_data = data[index1]
    v_labels = labels[index1]
    v_partial_labels = partial_labels[index1]


    return v_data, v_labels, v_partial_labels

def real_data_load(dataname, batch_size):
    # 加载初始数据
    train_X, train_Y, train_p_Y, test_X, test_Y = next(extract_data(dataname))

    # 创建全局数据集
    global_dataset = GlobalDataset(train_X, train_p_Y, train_Y)

    # 创建静态训练集和测试集
    train_loader = create_train_loader(train_X, train_p_Y, train_Y, batch_size=batch_size)
    test_loader = create_train_loader(test_X, test_Y, test_Y, batch_size=batch_size)

    return train_loader, test_loader, global_dataset

 # #随机选1/10
    # num_samples = len(data)
    # num_validation = num_samples // 5
    #
    # # 生成随机索引
    # random_indices = torch.randperm(num_samples)[:num_validation]
    #
    # # 选择验证集
    # v_data = data[random_indices]
    # v_labels = labels[random_indices]
    # v_partial_labels = partial_labels[random_indices]