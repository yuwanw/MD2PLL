# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io
import torch

import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial.distance

import torch.nn.functional as F




def load_matlab_file(path_file, name_field):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containig the field name

    """
    f = scipy.io.loadmat(path_file)
    out = f[name_field]
    return out


def load_data_monti(dataset):
    """
    Loads data
    """
    path_dataset = r'C:/Users/93613/Desktop/DATA/' + dataset + '.mat'
    #path_dataset = 'D:/TJuniversity/machine_learning/1/PLGAN/data/' + dataset + '.mat'

    print("\033[34mload %s data.....\033[0m" %dataset)

    data = load_matlab_file(path_dataset, 'data')
    partial_target = load_matlab_file(path_dataset, 'partial_target')
    target = load_matlab_file(path_dataset, 'target')

    #data = normalize_features(data)

    num_instances = data.shape[0]
    num_labels = target.shape[0]

    print('number of users = ', num_instances)
    print('number of item = ', num_labels)

    return data,partial_target,target


def K_Fold_CV(Data, k):
    """
    k-fold Cross-validation

    Parameter Description：
    Data：Given data set
    k：Fold
    """
    data_num = Data.shape[0]
    data_idx = list(range(data_num))
    np.random.shuffle(data_idx)
    test_num = Data.shape[0] // k
    train = []
    test = []

    for i in range(k):
        print('data processing, Cross validation {:4d}'.format(i+1))
        train_idx = data_idx.copy()
        start_ind = i*test_num
        if start_ind + test_num < data_num:
           test.append(data_idx[start_ind:(start_ind + test_num)])
           del train_idx[start_ind:(start_ind + test_num)]
        else:
           test.append(data_idx[start_ind:])
           del train_idx[start_ind:]
        train.append(train_idx)
    return train,test




def cluster_center(data, partial_labels, axis=0):
    """Return the cluster center."""
    center = []
    deg_label = np.sum(partial_labels, axis=axis)
    real_index = np.where(deg_label == 1)
    real_ins = data[real_index]
    real_label = np.argmax(partial_labels[real_index], axis=axis)
    for i in range(partial_labels.shape[1]):
        index = np.where(real_label == i)
        num_index = len(index[0])
        if num_index != 0:
            ins_center = real_ins[index].mean(axis=0)
            center.append(ins_center)
        else:
            index1 = np.where(partial_labels[:, i] == 1)
            ins_center = data[index1].mean(axis=0)
            center.append(ins_center)
    return center


def att_cos_dis(target, center):
    attention_distribution = []
    for i in range(center.size(0)):
        attention_score = torch.cosine_similarity(target, center[i], dim=0)
        attention_distribution.append(attention_score)

    return 0.5 + 0.5*np.array(attention_distribution)


def att_dis(target, center):
    attention_distribution = []
    for i in range(center.size(0)):
        attention_score = torch.dist(target, center[i], p=2)
        attention_distribution.append(attention_score)
    return data_normal(np.array(attention_distribution))

def data_normal(data):
    d_min = data.min()
    if d_min < 0:
        data += torch.abs(d_min)
        d_min = data.min()
    d_max = data.max()
    dst = d_max - d_min
    return (data-d_min)/dst

def partial_loss(output1, target,dis_label):
    output = F.softmax(output1, dim=1)
    l = dis_label*target * torch.log(output)
    loss = (-torch.sum(l)) / l.size(0)

    revisedY = target.clone()
    revisedY[revisedY > 0]  = 1
    revisedY = revisedY * output
    revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)

    new_target = revisedY
    return loss, new_target

def partial_loss_ablation(output1, target, dis_label):
    output = F.softmax(output1, dim=1)
    l = dis_label  * torch.log(output)
    loss = (-torch.sum(l)) / l.size(0)
    return loss, target.clone()  # 直接返回原始标签，不修正

def center_sim(data, center):
    sim_distribution = []
    for i in range(data.size(0)):
        cos_sim = att_cos_dis(data[i], center)
        sim_distribution.append(cos_sim)
    sim_dis = torch.Tensor(sim_distribution)

    return sim_dis

def updat_center(data, sim_dis, center, alpha):

    _, sim_fea = torch.max(sim_dis.data, 1)
    for i in range(center.shape[0]):
        index = np.where(sim_fea == i)
        ins_center = data[index].mean(axis=0)
        center[i] = center[i] - alpha*(center[i]-ins_center)
    return torch.Tensor(center)

def disCosine(x,y):
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    #dist = torch.matmul(x.float().detach(),y.transpose(0,1).float().detach())

    dist = torch.matmul(x.float(), y.transpose(0, 1).float())



    # xx = torch.sum(x**2, dim=1)**0.5
    # x = x/xx[:,np.newaxis].double()
    # yy = torch.sum(y**2, dim=1)**0.5
    # y = y/yy[:,np.newaxis].double()


    #dist = torch.mm(a,b.transpose(0,1))
    return 0.5 + 0.5*dist
