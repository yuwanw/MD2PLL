import os
import os.path
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import argparse
import numpy as np
import pandas as pd
import time
from utils.utils_algo import DataIterator
from models import *
from meta_models import *
from mlc_utils import clone_parameters, tocuda, DummyScheduler
from collections import deque
from tqdm import tqdm
from mlc import step_hmlc_K
from sklearn.mixture import GaussianMixture
from utils.utils_loss import partial_loss
from utils.models import linear, mlp
from datasets.mnist import mnist
from datasets.fashionmnist import *
from datasets.v_real_data import *
from datasets.cifar10 import *
from utils.utils_k import *
from kvit import k_max_vit
import torch,gc
from resnet import *
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import collections
from sklearn.calibration import calibration_curve

torch.manual_seed(0);
torch.cuda.manual_seed_all(0)

parser = argparse.ArgumentParser(
    prog='PLL_MLC demo file.',
    usage='Demo with partial labels.',
    description='A simple demo file with cifar10 dataset.',
    epilog='end',
    add_help=True)

parser = argparse.ArgumentParser()
parser.add_argument('-ds', help='specify a dataset', type=str, default='lost',
                    choices=['mnist', 'fashion', 'kmnist', 'cifar10','lost','Soccer Player','MSRCv2','FG_NET','BirdSong','Yahoo! News'], required=False)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--data_seed', type=int, default=1)

parser.add_argument('--epochs', '-e', type=int, default=150, help='Number of epochs to train.')

parser.add_argument('--num_iterations', default=100000, type=int)
parser.add_argument('--every', default=100, type=int, help='Eval interval (default: 100 iters)')
parser.add_argument('--bs', default=100, type=int, help='batch size')
parser.add_argument('--test_bs', default=64, type=int, help='batch size')
parser.add_argument('--gold_bs', type=int, default=32)
parser.add_argument('--cls_dim', type=int, default=64, help='Label embedding dim (Default: 64)')
parser.add_argument('--grad_clip', default=0.0, type=float, help='max grad norm (default: 0, no clip)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for optimizer')
parser.add_argument('--main_lr', default=0.1, type=float, help='lr for main net')
parser.add_argument('--meta_lr', default=3e-4, type=float, help='lr for meta net')
parser.add_argument('--optimizer', default='sgd', type=str, choices=['adam', 'sgd', 'adadelta'])
parser.add_argument('--opt_eps', default=1e-8, type=float, help='eps for optimizers')
#parser.add_argument('--tau', default=1, type=float, help='tau')
parser.add_argument('--wdecay', default=4e-4, type=float, help='weight decay (default: 5e-4)')

# noise parameters
parser.add_argument('--corruption_type', default='unif', type=str, choices=['unif', 'flip'])
parser.add_argument('--corruption_level', default='0.4', type=float, help='Corruption level')
parser.add_argument('--gold_fraction', default='0.02', type=float, help='Gold fraction')

parser.add_argument('--skip', default=False, action='store_true', help='Skip link for LCN (default: False)')
parser.add_argument('--sparsemax', default=False, action='store_true', help='Use softmax instead of softmax for meta model (default: False)')
parser.add_argument('--tie', default=False, action='store_true', help='Tie label embedding to the output classifier output embedding of metanet (default: False)')

parser.add_argument('--runid', default='exp', type=str)
parser.add_argument('--queue_size', default=1, type=int, help='Number of iterations before to compute mean loss_g')

############## LOOK-AHEAD GRADIENT STEPS FOR MLC ##################
parser.add_argument('--gradient_steps', default=1, type=int, help='Number of look-ahead gradient steps for meta-gradient (default: 1)')

# CIFAR
# Positional
parser.add_argument('--data_path', default='data', type=str, help='Root for the datasets.')
# Optimization options
parser.add_argument('--nosgdr', default=False, action='store_true', help='Turn off SGDR.')

# Acceleration
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')
# i/o
parser.add_argument('--logdir', type=str, default='runs', help='Log folder.')
parser.add_argument('--local_rank', type=int, default=-1, help='local rank (-1 for local training)')




parser.add_argument('-lr', help='optimizer\'s learning rate', type=float, default=1e-3)
parser.add_argument('-wd', help='weight decay', type=float, default=1e-5)




parser.add_argument('-model', help='model name', type=str, default='mlp',
                    choices=['linear', 'mlp', 'convnet', 'resnet'], required=False)
parser.add_argument('-decaystep', help='learning rate\'s decay step', type=int, default=500)
parser.add_argument('-decayrate', help='learning rate\'s decay rate', type=float, default=1)

parser.add_argument('-fake_number', help='flipping strategy', type=int, default=2)
parser.add_argument('-partial_rate', help='flipping probability', type=float, default=0.5)
parser.add_argument('-partial_type', help='flipping probability', type=str, default='binomial')

parser.add_argument('-dir', help='result save path', type=str, default='results/', required=False)
parser.add_argument('-alpha', help='alpha', type=float, default=1e-4)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_epoch = -1



# cuda set up
# torch.cuda.set_device(0) # local GPU

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True




# loss function for hard label and soft labels
hard_loss_f = F.cross_entropy
soft_loss_f = partial_loss
criterion = nn.CrossEntropyLoss(reduction='none')




# load dataset
if args.ds == 'fashion':
    num_features = 28 * 28
    num_classes = 10
    num_training = 60000
    global_dataset, train_loader, test_loader= load_fashionmnist(partial_rate=args.partial_rate, batch_size=args.bs)
if args.ds == 'cifar10':
    input_channels = 3
    num_classes = 10
    dropout_rate = 0.25
    num_training = 50000
    global_dataset, train_loader, test_loader = load_cifar10(partial_rate=args.partial_rate, batch_size=args.bs)
if args.ds == 'lost':
    num_classes = 16
    num_training = 1122
    train_loader, test_loader, global_dataset = real_data_load(dataname=args.ds,batch_size=args.bs)
if args.ds == 'Soccer Player':
    num_classes = 171
    num_training = 17472
    train_loader, test_loader, global_dataset = real_data_load(dataname=args.ds,batch_size=args.bs)
if args.ds == 'FG_NET':
    num_classes = 78
    num_training = 1002
    train_loader, test_loader, global_dataset = real_data_load(dataname=args.ds,batch_size=args.bs)
if args.ds == 'MSRCv2':
    num_classes = 23
    num_training = 1758
    train_loader, test_loader, global_dataset = real_data_load(dataname=args.ds,batch_size=args.bs)
if args.ds == 'BirdSong':
    num_classes = 13
    num_training = 4998
    train_loader, test_loader, global_dataset = real_data_load(dataname=args.ds,batch_size=args.bs)
if args.ds == 'Yahoo! News':
    num_classes = 219
    num_training = 22991
    train_loader, test_loader, global_dataset = real_data_load(dataname=args.ds,batch_size=args.bs)



# learning rate
lr_plan = [args.lr] * args.epochs
for i in range(0, args.epochs):
    lr_plan[i] = args.lr * args.decayrate ** (i // args.decaystep)


def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_plan[epoch]


# result dir
save_dir = './' + args.dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_file = os.path.join(save_dir,
                         ('KmaxVIT' + args.ds + '_' + str(args.fake_number) + '_' + str(args.partial_rate) + '.txt'))


def build_models(dataset, num_classes):
    cls_dim = args.cls_dim  # input label embedding dimension

    if dataset in ['cifar10', 'cifar100', 'fashion']:
        from datasets.resnet import resnet32
        model = resnet32(num_classes)
        main_net = model
        hx_dim = 64  # 0 if isinstance(model, WideResNet) else 64 # 64 for resnet-32
        # meta_net = MetaNet(hx_dim, cls_dim, 128, num_classes, args)
        meta_net = MetaNet(hx_dim, cls_dim, 128, num_classes, args)
    if dataset in ['lost']:
        from utils.models import mlp
        model = mlp(n_inputs=108,n_outputs=16,num_classes=num_classes)
        main_net = model
        # main net
        # meta net
        hx_dim = 16  # 0 if isinstance(model, WideResNet) else 64 # 64 for resnet-32
        meta_net = MetaNet(hx_dim, cls_dim, 128, num_classes, args)
    if dataset in ['Soccer Player']:
        from utils.models import mlp
        model = mlp(n_inputs=279,n_outputs=171,num_classes=num_classes)
        main_net = model
        # main net
        # meta net
        hx_dim = 171  # 0 if isinstance(model, WideResNet) else 64 # 64 for resnet-32
        meta_net = MetaNet(hx_dim, cls_dim, 128, num_classes, args)
    if dataset in ['MSRCv2']:
        from utils.models import mlp
        model = mlp(n_inputs=48,n_outputs=23,num_classes=num_classes)
        main_net = model
        # main net
        # meta net
        hx_dim = 23  # 0 if isinstance(model, WideResNet) else 64 # 64 for resnet-32
        meta_net = MetaNet(hx_dim, cls_dim, 128, num_classes, args)
    if dataset in ['BirdSong']:
        from utils.models import mlp
        model = mlp(n_inputs=38, n_outputs=13, num_classes=num_classes)
        main_net = model
        # main net
        # meta net
        hx_dim = 13  # 0 if isinstance(model, WideResNet) else 64 # 64 for resnet-32
        meta_net = MetaNet(hx_dim, cls_dim, 128, num_classes, args)
    if dataset in ['Yahoo! News']:
        from utils.models import mlp
        model = mlp(n_inputs=163, n_outputs=219, num_classes=num_classes)
        main_net = model
        hx_dim = 219  # 0 if isinstance(model, WideResNet) else 64 # 64 for resnet-32
        meta_net = MetaNet(hx_dim, cls_dim, 128, num_classes, args)
    if dataset in ['FG_NET']:
        from utils.models import mlp
        model = mlp(n_inputs=262, n_outputs=78, num_classes=num_classes)
        main_net = model
        hx_dim = 78  # 0 if isinstance(model, WideResNet) else 64 # 64 for resnet-32
        meta_net = MetaNet(hx_dim, cls_dim, 128, num_classes, args)





    main_net = main_net
    meta_net = meta_net

    return main_net, meta_net


def setup_training(main_net, meta_net):

    # ============== setting up from scratch ===================
    # set up optimizers and schedulers
    # meta net optimizer
    optimizer = torch.optim.Adam(meta_net.parameters(), lr=args.meta_lr,
                                 weight_decay=0, #args.wdecay, # meta should have wdecay or not??
                                 amsgrad=True, eps=args.opt_eps)
    scheduler = DummyScheduler(optimizer)

    # main net optimizer
    main_params = main_net.parameters()

    if args.optimizer == 'adam':
        main_opt = torch.optim.Adam(main_params, lr=args.main_lr, weight_decay=args.wdecay, amsgrad=True, eps=args.opt_eps)
    elif args.optimizer == 'sgd':
        main_opt = torch.optim.SGD(main_params, lr=args.main_lr, weight_decay=args.wdecay, momentum=args.momentum)

    if args.ds in ['cifar10', 'cifar100']:
        # follow MW-Net setting
        main_schdlr = torch.optim.lr_scheduler.MultiStepLR(main_opt, milestones=[80,100], gamma=0.1)
    elif args.ds in ['clothing1m']:
        main_schdlr = torch.optim.lr_scheduler.MultiStepLR(main_opt, milestones=[5], gamma=0.1)
    else:
        main_schdlr = DummyScheduler(main_opt)

    last_epoch = -1

    return main_net, meta_net, main_opt, optimizer, main_schdlr, scheduler, last_epoch


def binarize_class(y):
    label = y.reshape(len(y), -1)
    enc = OneHotEncoder(categories='auto')
    enc.fit(label)
    label = enc.transform(label).toarray().astype(np.float32)

    return label


from matplotlib import cm

import numpy as np
import matplotlib.pyplot as plt
import collections

import numpy as np
import matplotlib.pyplot as plt
import collections


def compute_ece(confidences, correct, M=10):
    """Compute ECE with FIXED binning (aligned with calibration_graph_new)"""
    confidences = np.asarray(confidences)
    correct = np.asarray(correct)
    N = len(confidences)

    # Fixed-width bins (0.0, 0.1, ..., 1.0)
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2  # Midpoints (0.05, 0.15, ...)

    ece = 0.0
    mce = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    print("\n**ECE Bin Breakdown (Fixed Bins, width=0.1)**")
    print(f"{'Bin':<5} {'Range':<20} {'Count':<8} {'Avg Confidence':<15} {'Accuracy':<10} {'Gap':<10}")

    for i in range(M):
        in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        bin_count = np.sum(in_bin)
        bin_counts.append(bin_count)

        if bin_count > 0:
            bin_acc = np.mean(correct[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            gap = abs(bin_acc - bin_conf)
        else:
            bin_acc = bin_conf = gap = 0.0  # Handle empty bins

        bin_accs.append(bin_acc)
        bin_confs.append(bin_conf)
        ece += (bin_count / N) * gap
        mce = max(mce, gap)

        # Print bin stats (same format as calibration_graph_new)
        print(
            f"{i + 1:<5} [{bin_boundaries[i]:.3f}, {bin_boundaries[i + 1]:.3f})  {bin_count:<8} {bin_conf:<15.4f} {bin_acc:<10.4f} {gap:<10.4f}")

    return ece, mce, bin_centers, bin_accs, bin_confs, bin_counts


def calibration_plot(preds, scores, true):
    """完全复现原始可靠性图的双柱状图设计"""
    # 计算指标（固定分箱）
    ece, mce, bin_centers, bin_accs, bin_confs, bin_counts = compute_ece(scores, (preds == true))
    avg_acc = np.mean(preds == true)
    avg_conf = np.mean(scores)

    # 生成分箱区间（与原始代码一致）
    interval = 0.1
    num_intervals = int(1 / interval)
    IntervalRange = collections.namedtuple('IntervalRange', 'start end')
    intervals = [IntervalRange(start=i * interval, end=(i + 1) * interval) for i in range(num_intervals)]

    # 准备绘图数据
    result = [conf - acc for acc, conf in zip(bin_accs, bin_confs)]  # 或 acc - conf
    plt.figure(figsize=(12, 6))

    # --- 1. 置信度分布直方图 ---
    plt.subplot(1, 2, 1)
    plt.bar([intv.start for intv in intervals], np.array(bin_accs),
            width=0.1, color='lightskyblue', edgecolor='silver', align='edge', alpha=0.6)
    plt.axvline(avg_acc, linestyle='--', color='b')
    plt.text(avg_acc - 0.08, 0.5, 'accuracy', rotation=90, color='b')
    plt.axvline(avg_conf, linestyle='--', color='g')
    plt.text(avg_conf + 0.03, 0.4, 'avg. confidence', rotation=90, color='g')
    plt.ylabel('Accuracy')
    plt.xlabel('confidence')
    plt.title('confidence histogram')
    plt.grid(True, alpha=0.7, linestyle='--')

    # --- 2. 可靠性图（完全复现原始设计）---
    plt.subplot(1, 2, 2)
    # 灰色对角线
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")

    # 双柱状图（关键修复部分）
    plt.bar([intv.start for intv in intervals], result,
            bottom=bin_accs, width=0.1, color='coral', edgecolor='orangered',
            hatch='//', align='edge', alpha=0.5, label='Ideal')
    plt.bar([intv.start for intv in intervals], bin_accs,
            width=0.1, color=plt.get_cmap('bwr')(np.linspace(0.4, 0, 10)),
            edgecolor='black', align='edge', alpha=0.9, label='Outputs')

    # 点线图
    plt.plot(bin_centers, bin_accs, marker="o", linestyle="-", color="red", label="Accuracy")
    plt.plot(bin_centers, bin_confs, marker="s", linestyle="--", color="blue", label="Confidence")

    # 标注和样式
    plt.text(0.6, 0.05, f'ECE={ece:.3f}', backgroundcolor='w', alpha=0.6)
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('confidence')
    plt.title('reliability diagram')

    plt.tight_layout()
    plt.show()



def test(model, test_loader,save_plot=False, epoch=None):  # this could be eval or test
    # //////////////////////// evaluate method ////////////////////////

    all_confidences = []
    all_predictions = []
    all_labels = []
    all_output = []  # 初始化
    all_is_correct = []
    correct = 0
    nsamples = 0
    # forward
    model.eval()

    # for idx, (data, target, trues, indexes) in enumerate(test_loader):
    for idx, (data, target, trues, indexes) in enumerate(test_loader):#, trues, indexes
        # forward
        with torch.no_grad():
            output = model(data,return_h=False)#main
            # output = model(data,target) #meta model
            output = F.softmax(output, dim=1)
            confidences, pred = torch.max(output.data, 1)
            _, labels = torch.max(target, 1)#
            nsamples += data.size(0)
            correct += (pred == labels).sum().item()#
            all_confidences.extend(confidences.cpu().numpy())
            all_output.extend(output.cpu().numpy())
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_is_correct.extend((pred == labels).cpu().numpy().astype(int))
            # accuracy

    confidences = np.array(all_confidences)
    predictions = np.array(all_predictions)
    true_labels = np.array(all_labels)
    output = np.array(all_output)
    if save_plot:
        calibration_plot(predictions, confidences, true_labels)
    test_acc = 100 * float(correct) / float(nsamples)
    model.train()

    return test_acc



def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)  # 避免除零


# 设置Seaborn风格（学术风格）
sns.set_style("whitegrid")  # 白色背景+网格线
sns.set_palette("pastel")   # 柔和配色


def main():
    # print ('loading dataset...')

    vnet = VNet(1, 100, 1)  # .cuda()
    vnet1 = VNet(1, 100, 1)  # .cuda()
    # //////////////////////// build main_net and meta_net/////////////
    main_net, meta_net = build_models(args.ds, num_classes)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main_net, meta_net, main_opt, optimizer, main_schdlr, scheduler, last_epoch = setup_training(main_net, meta_net)

    # //////////////////////// switching on training mode ////////////////////////
    meta_net.train()
    main_net.train()

    # set done
    best_test_acc = 0
    best_val_acc = 0
    best_epoch = -1  # 记录最佳轮次编号（初始-1表示无效值）
    best_clean = None
    best_noisy = None

    args.dw_prev = [0 for param in meta_net.parameters()]  # 0 for previous iteration
    args.steps = 0

    # v_data, v_partialY, v_true_labels = select_validation_data(global_dataset)
    # valid_dataset = ValidAugmentation(v_data, v_partialY, v_true_labels)
    # valid_loader = data.DataLoader(valid_dataset, batch_size=100, shuffle=False, num_workers=4)

    v_data, v_partialY, v_true_labels = select_validation_data(global_dataset)
    valid_loader = create_train_loader(v_data, v_partialY, v_true_labels, batch_size=args.bs)

    for epoch in tqdm(range(last_epoch+1, args.epochs)):# change to epoch iteration
        l = []
        indexes_list = []  # 存储每个 batch 的 indexes
        new_labels_list = []
        for i, (data_s, target_s, trues, indexes) in enumerate(train_loader):# _,
            data_g, target_g, trues_g, indexes_g = next(iter(valid_loader))
            # bi-level optimization stage
            eta = main_schdlr.get_lr()[0]
            loss, new_label = step_hmlc_K(main_net, main_opt, hard_loss_f,
                                            meta_net, optimizer, partial_loss_ablation,
                                            data_s, target_s, data_g, target_g,
                                            eta, args, vnet, criterion)
            # l = torch.cat(losses)
            l.append(loss)
            indexes_list.append(indexes)
            new_labels_list.append(new_label.cpu().detach())

        l = torch.cat(l)
        l = (l - l.min()) / (l.max() - l.min())
        input_loss = l.reshape(-1, 1)
        input_loss = input_loss.detach().numpy()
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        prob = prob[:, gmm.means_.argmin()]
        for batch_indexes, batch_new_labels in zip(indexes_list, new_labels_list):
            for j, k in enumerate(batch_indexes):
                if prob[k] > 0.8:
                    sample_data = train_loader.dataset.train_X[k]#data train_X
                    sample_label = batch_new_labels[j, :]  # 使用更新后的标签
                    v_data.append(sample_data)
                    v_partialY.append(sample_label)
                    v_true_labels.append(sample_label)
                    # sample_data = train_loader.dataset.train_X[k].unsqueeze(0)  # 增加维度
                    # sample_label = batch_new_labels[j, :].unsqueeze(0)
                    # v_data = torch.cat([v_data, sample_data], dim=0)  # 合并张量
                    # v_partialY = torch.cat([v_partialY, sample_label], dim=0)
                    # v_true_labels = torch.cat([v_true_labels, sample_label], dim=0)
        valid_loader = create_train_loader(v_data, v_partialY, v_true_labels, batch_size=100)

        # if epoch == 80:  # 指定epoch
        #     # 分离两类数据
        #     high_conf_idx = np.where(prob > 0.8)[0]
        #     low_conf_idx = np.where(prob <= 0.8)[0]
        #     high_conf_loss = input_loss[high_conf_idx].flatten()
        #     low_conf_loss = input_loss[low_conf_idx].flatten()
        #
        #     # 绘制概率密度分布
        #     plt.figure(figsize=(10, 6))
        #     sns.kdeplot(high_conf_loss, color="green", label="High Confidence (prob > 0.8)", fill=True)
        #     sns.kdeplot(low_conf_loss, color="red", label="Low Confidence (prob <= 0.8)", fill=True, alpha=0.6)
        #     plt.xlabel("Normalized Loss Value")
        #     plt.ylabel("Probability Density")
        #     plt.legend()
        #     plt.title(f"GMM-based Loss Distribution (Epoch {epoch})")
        #     sns.despine()
        #     plt.savefig(f"gmm_epoch_{epoch}.png")  # 保存图像
        #     plt.close()
        #
        #     # 导出数据到文件
        #     np.savez(f"gmm_results_epoch_{epoch}.npz",
        #              high_conf_indices=high_conf_idx,
        #              low_conf_indices=low_conf_idx,
        #              high_conf_loss=high_conf_loss,
        #              low_conf_loss=low_conf_loss,
        #              gmm_means=gmm.means_,
        #              gmm_covariances=gmm.covariances_)

        # valid_dataset = ValidAugmentation(v_data, v_partialY, v_true_labels)
        # valid_loader = data.DataLoader(valid_dataset, batch_size=100, shuffle=False, num_workers=4)

        # evaluation on validation set
        print('evaluating model...')
        # val_acc = test(main_net, test_loader)
        if epoch == args.epochs - 1:
            val_acc = test(main_net, test_loader, save_plot=True, epoch=epoch)
        else:
            val_acc = test(main_net, test_loader)


        if val_acc > best_val_acc:
            best_val_acc = val_acc
        print(
            ' TRAIN: epoch = {:.4f}\t : Best Acc.: = {:.4f}\t, Test Acc.: = {:.4f}\n'.format(
                epoch, best_val_acc, val_acc))#, ece , ECE.: = {:.4f}

        with open(save_file, 'a') as file:
            file.write(str(int(epoch)) + ': Best Acc.: ' + str(round(best_val_acc, 4)) + ' , Test Acc.: ' + str(
                round(val_acc, 4)) + '\n')
        # np.savez_compressed(
        #     f'best_samples_epoch{best_epoch}.npz',  # 文件名包含epoch
        #     clean_data=np.array(best_clean),  # 保存为float32减少空间
        #     noisy_data=np.array(best_noisy),
        #     clean_probs=prob[prob > 0.8],  # 可选：保存对应的概率值
        #     noisy_probs=prob[prob <= 0.8],
        #     meta={
        #         'epoch': best_epoch,
        #         'val_acc': float(best_val_acc),  # 保存关键元数据
        #         'normalized': False  # 标记是否已归一化
        #     }
        # )

        # # 同时保存为CSV（适合表格数据）
        # pd.DataFrame({
        #     'feature_value': np.concatenate([best_clean, best_noisy]),
        #     'type': ['clean'] * len(best_clean) + ['noisy'] * len(best_noisy),
        #     'prob': np.concatenate([prob[prob > 0.8], prob[prob <= 0.8]])
        # }).to_csv(f'best_samples_epoch{best_epoch}.csv', index=False)

    # # 绘制分布对比图
    # plt.figure(figsize=(10, 6), dpi=100)  # 增大画布尺寸和分辨率
    #
    # # 绘制Clean数据分布（绿色系）
    # sns.histplot(best_clean, color="limegreen", label="High-confidence (clean)",
    #              kde=True, bins=30, edgecolor='darkgreen', linewidth=0.8,
    #              kde_kws={'linewidth': 2})
    #
    # # 绘制Noisy数据分布（红色系）
    # sns.histplot(best_noisy, color="salmon", label="Low-confidence (noisy)",
    #              kde=True, bins=30, alpha=0.5, edgecolor='darkred', linewidth=0.8,
    #              kde_kws={'linewidth': 2, 'linestyle': '--'})
    #
    # # 标注统计量
    # plt.axvline(x=np.mean(best_clean), color='darkgreen',
    #             linestyle=':', linewidth=2, label=f'Clean Mean: {np.mean(best_clean):.2f}')
    # plt.axvline(x=np.mean(best_noisy), color='darkred',
    #             linestyle=':', linewidth=2, label=f'Noisy Mean: {np.mean(best_noisy):.2f}')
    # # 标注高斯组分（可选）
    # plt.axvline(x=np.mean(best_clean), color='darkgreen', linestyle='--', linewidth=1, label='Clean Mean')
    # plt.axvline(x=np.mean(best_noisy), color='darkred', linestyle='--', linewidth=1, label='Noisy Mean')
    # # 坐标轴和标题说明
    # plt.xlabel("Feature Value (Normalized)",
    #            fontsize=12, fontweight='bold', labelpad=10)  # 横轴：归一化后的特征值
    # plt.ylabel("Probability Density",
    #            fontsize=12, fontweight='bold', labelpad=10)  # 纵轴：概率密度
    # plt.title("High-confidence vs Low-confidence Samples Distribution\n"
    #           f"(Best Epoch: {best_epoch}, Test Acc: {best_val_acc:.2%})",
    #           fontsize=14, fontweight='bold', pad=20)
    #
    # # 图例和样式美化
    # legend = plt.legend(frameon=True, shadow=True,
    #                     fontsize=11, title="Sample Category",
    #                     title_fontsize=12, bbox_to_anchor=(1.05, 1),
    #                     loc='upper left')
    # legend.get_frame().set_facecolor('#F5F5F5')
    #
    # # 网格和边框调整
    # plt.grid(True, linestyle='--', alpha=0.3)
    # sns.despine(left=True, bottom=True)
    # plt.tight_layout()
    #
    # # 保存高质量图片
    # plt.savefig(f'clean_vs_noisy_epoch{best_epoch}.png',
    #             dpi=300, bbox_inches='tight', transparent=False)
    # plt.show()



    return best_test_acc









if __name__ == '__main__':
    main()


