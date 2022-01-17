# coding=utf-8
# transductive learning with Graph Attention Network
# 18 feature parameters
# writer: Owens

import os
import argparse
import time
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy import io
from sklearn.model_selection import train_test_split
from models.model_LSTM import *
from models.model_GAT import GAT, SpGAT
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from summary import evaluate_metrics
import scipy.sparse as sp
import torch.nn.functional as F
from collections import Counter
import random


def data_loader(fea_dir):
    print("===loading feature matrix===")
    data = []
    labels = []
    for mat_idx in os.listdir(fea_dir):
        mat_name = fea_dir + "/" + mat_idx
        fea_loader = io.loadmat(mat_name)
        data.append(fea_loader['fea'])
        labels.append(fea_loader['labels'].squeeze())

    return data, labels


def run_model(model, optimizer, x, y, args):
    adj = gen_adj(y)
    epoch_num = len(x)
    offset_train = range(int(args.train_ratio * epoch_num))
    offset_val = range(int(args.train_ratio * epoch_num), int(args.val_ratio * epoch_num))
    offset_test = range(int(args.val_ratio * epoch_num), epoch_num-1)

    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = args.episode_num + 1
    best_episode = 0

    feature = Variable(torch.tensor(x), requires_grad=False).float()  # [T, 18]
    target = Variable(torch.tensor(y), requires_grad=False).long()  # [T, 5]
    adj = Variable(torch.tensor(adj.todense()), requires_grad=False).float()  # [T, T]
    print("normalized adj=", adj)
    model.zero_grad()
    for episode in range(args.episode_num):
        print("@@@ Episode: {}/{}".format(episode+1, args.episode_num))
        loss_values.append(train_episode(model, optimizer, feature, target, adj, episode, offset_train, offset_val, args.fastmode))

        torch.save(model.state_dict(), '{}.pkl'.format(episode))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_episode = episode
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_episode:
                os.remove(file)
    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_episode:
            os.remove(file)
    print("\tOptimization Finished!")
    print("\tTotal time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    print('\tLoading {}th epoch'.format(best_episode))
    model.load_state_dict(torch.load('{}.pkl'.format(best_episode)))

    # draw loss figure
    plt.figure()
    plt.plot(loss_values)
    plt.show()

    compute_test(model, feature, target, adj, offset_test, args.classes)

    return model


def train_episode(model, optimizer, feature, target, adj, episode, offset_train, offset_val, fastmode):
    t = time.time()
    criterion = nn.NLLLoss()
    model.train()
    optimizer.zero_grad()
    output = model(feature, adj)
    loss_train = criterion(output[offset_train], target[offset_train])
    acc_train = accuracy(output[offset_train], target[offset_train])
    loss_train.backward()
    optimizer.step()

    if not fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(feature, adj)

    loss_val = criterion(output[offset_val], target[offset_val])
    acc_val = accuracy(output[offset_val], target[offset_val])
    print('Episode: {:04d}'.format(episode+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


# Generate the adjacency matrix
def gen_adj(labels):
    edges = []
    for epoch_i in range(len(labels)-1):
        # flags = np.zeros(5, dtype=int)
        epoch_j = epoch_i + 1
        # flags[labels[epoch_i]] = 1
        # flags[labels[epoch_j]] = 1
        edges.append([epoch_i, epoch_j])
        # epoch_j += 1
        # while np.min(flags) == 0 and epoch_j < len(labels)-1:
        #     if
        #     epoch_j += 1
    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def compute_test(model, feature, target, adj, offset_test, classes):
    model.eval()
    output = model(feature, adj)
    loss_test = F.nll_loss(output[offset_test], target[offset_test])
    acc_test = accuracy(output[offset_test], target[offset_test])
    print("\tTest set results:", "loss= {:.4f}".format(loss_test.data.item()), "accuracy= {:.4f}".format(acc_test.data.item()))

    y_pred = np.array(output[offset_test].max(1)[1].type_as(target))
    y_true = np.array(target[offset_test])

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    ck_score = cohen_kappa_score(y_true, y_pred)
    acc_avg, acc, f1_macro, f1, sensitivity, specificity, precision = evaluate_metrics(cm)

    print('Average Accuracy -> {:>6.4f}, Macro F1 -> {:>6.4f} and Cohen\'s Kappa -> {:>6.4f} on test set'.format(acc_avg, f1_macro, ck_score))
    for index_ in range(len(classes)):
        print("\t{} rhythm -> Sensitivity: {:1.4f}, Specificity: {:1.4f}, Precision: {:1.4f}, F1 : {:1.4f} Accuracy: {:1.4f}".format(
                classes[index_], sensitivity[index_], specificity[index_], precision[index_], f1[index_], acc[index_]))
    print("\tAverage -> Sensitivity: {:1.4f}, Specificity: {:1.4f}, Precision: {:1.4f}, F1-score: {:1.4f}, Accuracy: {:1.4f}".format(
            np.mean(sensitivity), np.mean(specificity), np.mean(precision), np.mean(f1), np.mean(acc)))


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--fea_dir', type=str, default='./preparation/fea_par/', help='feature matrix directory')
    parser.add_argument('--fea_dir', type=str, default='./preparation/subbands_fea/', help='feature matrix directory')
    parser.add_argument('--random_state', type=int, default=42, help='seed for random initialisation')
    parser.add_argument('--episode_num', type=int, default=1, help='number for training iteration')
    parser.add_argument('--train_ratio', type=float, default=0.1, help='the number of testing dataset')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='the number of testing dataset')
    # parser.add_argument('--test_ratio', type=float, default=0.3, help='the number of testing dataset')
    parser.add_argument('--input_size', '-is', type=int, default=18, help='feature vector size')
    parser.add_argument('--hidden_size', '-hs', type=int, default=128, help='sequence embedding size')
    parser.add_argument('--output_size', '-os', type=int, default=5, help='label class size')
    parser.add_argument('--score_mode', '-s', choices=['concat', 'double'], default='concat',
                        help='way to combine topics and scores')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay rate')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
    parser.add_argument('--dropout', type=float, default=0.8, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--nb_heads', type=int, default=5, help='Number of head attentions.')
    parser.add_argument('--patience', type=int, default=10, help='Patience')
    parser.add_argument('--classes', type=list, default=['W', 'N1', 'N2', 'N3', 'REM'], help='class names')

    args = parser.parse_args()

    classes = ['W', 'N1', 'N2', 'N3', 'REM']

    data, labels = data_loader(args.fea_dir)

    model = GAT(nfeat=args.input_size, nhid=args.hidden_size, nclass=len(classes), dropout=args.dropout,
                nheads=args.nb_heads, alpha=args.alpha)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # for seq_idx in range(len(data)):
    #     print("=====Sleep Sequence {}:{}=====".format(seq_idx+1, data[seq_idx].shape))
    #     model = train_model(model, optimizer, data[seq_idx], labels[seq_idx], args)
    seq_idx = 0
    print("=====Sleep Sequence {}:{}=====".format(seq_idx + 1, data[seq_idx].shape))
    run_model(model, optimizer, data[seq_idx], labels[seq_idx], args)

if __name__ == "__main__":
    main()
