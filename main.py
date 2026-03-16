# -*- coding: utf-8 -*-
import os
import os.path
import torch
from torch.autograd import Variable
import argparse
import time
from sklearn.preprocessing import StandardScaler
import scipy.io
from torch.utils.data import Dataset
from utils.utils_k import *
from kvit import k_max_vit

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



class ParitileLabelDate(Dataset):
    def __init__(self, data):
        self.data = data['data']
        self.labels = data['labels']
        self.trues = data['trues']
        self.indexes = data['indexes']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.trues[idx], self.indexes[idx]



torch.manual_seed(0); torch.cuda.manual_seed_all(0)

parser = argparse.ArgumentParser(
	prog='KMT-PLL demo file.',
	usage='Demo with partial labels.',
	description='A simple demo file with lost dataset.',
	epilog='end',
	add_help=True)

parser = argparse.ArgumentParser()
parser.add_argument('-lr', help='optimizer\'s learning rate', type=float, default=0.01)
parser.add_argument('-wd', help='weight decay', type=float, default=1e-5)
parser.add_argument('-bs', help='batch size', type=int, default=64)
parser.add_argument('-ep', help='number of epochs', type=int, default=100)
parser.add_argument('-decaystep', help='learning rate\'s decay step', type=int, default=500)
parser.add_argument('-decayrate', help='learning rate\'s decay rate', type=float, default=1)

parser.add_argument('-partial_type', help='flipping strategy', type=str, default='binomial', choices=['binomial', 'pair'])
parser.add_argument('-partial_rate', help='flipping probability', type=float, default=0.1)

parser.add_argument('-nw', help='multi-process data loading', type=int, default=0, required=False)
parser.add_argument('-dir', help='result save path', type=str, default='results/', required=False)
parser.add_argument("-d", "--dataset", type=str, default="lost",
                choices=['lost', 'Birdsong', 'ecoli_3_0.5', 'Msrsv2', 'FG_NET', 'segment_1_0.1','glass_1_0.1','movement_1_0.1'],
                help="Dataset string.")
parser.add_argument("-k", "--fold", type=int, default=10,
                help="k-fold Cross-validation")
parser.add_argument('-alpha', help='alpha', type=float, default=1e-4)


args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# learning rate
lr_plan = [args.lr] * args.ep
for i in range(0, args.ep):
    lr_plan[i] = args.lr * args.decayrate ** (i // args.decaystep)
def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_plan[epoch]


# result dir
save_dir = './' + args.dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_file = os.path.join(save_dir, (args.dataset + '.txt'))

# calculate accuracy
def evaluate(loader, model, am_center):
    model.eval()
    correct = 0
    total = 0
    for data, _, labels, _ in loader:
        images = data.to(device).float()
        labels = labels.to(device).float()
        output1,_ = model(images, am_center)
        output = F.softmax(output1, dim=1)
        _, pred = torch.max(output.data, 1)
        total += images.size(0)
        correct += (pred == labels).sum().item()
    acc = 100 * float(correct) / float(total)
    return acc





def main():
    print ('loading dataset...')
    data, partial_target, target = load_data_monti(args.dataset)
    train_idx, test_idx = K_Fold_CV(data,args.fold)
    num_features = data.shape[1]
    num_classes = partial_target.shape[0]


    """normalize feats: z = (x - u) / s"""
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)

    """Compute cluster center"""
    in_center = cluster_center(data, np.transpose(partial_target).toarray(), axis=1)



    for k in range(0,args.fold):

        train_dataset= {
           'data':data[train_idx[k]],
           'labels': np.transpose(partial_target[:,train_idx[k]]).toarray(),
           'trues': np.argmax(np.transpose(target[:,train_idx[k]]).toarray(), axis=1),
           'indexes': list(range(len(train_idx[k])))}

        test_dataset = {
            'data': data[test_idx[k]],
            'labels': np.transpose(partial_target[:, test_idx[k]]).toarray(),
            'trues': np.argmax(np.transpose(target[:, test_idx[k]]).toarray(), axis=1),
            'indexes': list(range(len(test_idx[k])))}

        train_dataset = ParitileLabelDate(train_dataset)
        test_dataset = ParitileLabelDate(test_dataset)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=args.bs,
                                                   num_workers=args.nw,
                                                   drop_last=True,
                                                   shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=args.bs,
                                                  num_workers=args.nw,
                                                  drop_last=False,
                                                  shuffle=False)


        print('building model...')

        net = k_max_vit(num_features, num_classes)
        net.to(device)
        print(net.parameters)

        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
        time_start = time.time()

        am_center = torch.tensor(in_center).float().to(device)

        best_test_acc = 0
        print('training...')


        for epoch in range(0, args.ep):
            net.train()
            adjust_learning_rate(optimizer, epoch)

            for i, (image, labels, trues, indexes) in enumerate(train_loader):
                images = Variable(image).to(device).float()
                labels = Variable(labels).to(device).float()
                trues = trues.to(device).float()
                output,am_center1= net(images, am_center)

                am_center = am_center - args.alpha * (am_center - am_center1)

                #dis_label = center_sim(data, am_center)
                dis_label = disCosine(images, am_center)

                loss, new_label = partial_loss(output, labels, trues, dis_label)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                # update weights

                for j, k in enumerate(indexes):
                    train_loader.dataset.labels[k, :] = new_label[j, :].detach().cpu().numpy()

                # am_center = updat_center(data, new_label, am_center, 0.01)


            print ('evaluating model...')
            train_acc = evaluate(train_loader, net, am_center)
            test_start = time.time()
            test_acc = evaluate(test_loader, net, am_center)
            test_end = time.time()

            if test_acc > best_test_acc:
                best_test_acc = test_acc

            print(
                ' TRAIN: epoch = {:.4f}\t : Training Acc.: = {:.4f}\t, Test Acc.: = {:.4f}\t, Test Time.: = {:.4f}\n'.format(
                    epoch, train_acc, test_acc, test_end-test_start))


            with open(save_file, 'a') as file:
                file.write(str(int(epoch)) + ': Training Acc.: ' + str(round(train_acc, 4)) + ' , Test Acc.: ' + str(
                    round(test_acc, 4)) + '\n')



        time_end = time.time()
        print('totally cost', time_end - time_start)
        print('Beat test acc.:= {:.4f}\t'.format(best_test_acc))





if __name__=='__main__':
    main()