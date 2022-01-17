# coding=utf-8
# run with classical machine learning methods
# 18 feature parameters
# writer: Owens

import os
import argparse
import numpy as np
from scipy import io
import torch
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from models.LeNet import *
from models.AlexNet import *
from models.ResNet import *
from models.DenseNet import *
from models.SE_DenseNet import *
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from summary import evaluate_metrics

def prepare_imf_bis(args):
    if os.path.exists('imf_bis_list.npz') is False:
        print("===loading IMF's bispectrum===")
        x = []
        y = np.zeros((0, 1))
        for subject_idx in os.listdir(args.imf_bis_dir):
            print("append subject:" + subject_idx)
            path_read = args.imf_bis_dir + subject_idx + '/'
            for epoch_idx in os.listdir(path_read):
                # print("epoch_idx=", epoch_idx)
                mat_name = path_read + epoch_idx
                bis_loader = io.loadmat(mat_name)

                amp = np.array(bis_loader['bs_amp'])
                pha = np.array(bis_loader['bs_pha'])

                label = np.array(bis_loader['label'])

                path_store = args.bis_npz_dir + subject_idx + '/'
                address = path_store + str(epoch_idx[:-4]) + '.npy'
                # print("address=", address)
                x.append(address)
                y = np.append(y, label, axis=0)
                bis = np.append(amp, pha).reshape((8, 256, 256))
                np.save(address, bis)

        np.savez('imf_bis_list.npz', x=x, y=y)
    else:
        print('imf_bis_list is already exist.')


class MyDataset(Dataset):
    def __init__(self, list, labels):
        self.list = list
        self.labels = labels

    def __getitem__(self, index):#返回的是tensor
        bis = np.load(self.list[index])
        target = self.labels[index]
        return bis, target

    def __len__(self):
        return len(self.list)


def list_loader(args):
    data = np.load('imf_bis_list.npz')
    x = data['x']
    y = data['y'].flatten()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=args.test_size, random_state=args.random_state)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=args.val_size, random_state=args.random_state)
    return x_train, y_train, x_val, y_val, x_test, y_test


def main_ml():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imf_bis_dir', type=str, default='./preparation/imf_bis/', help='IMF bispectrum matrix directory as .mat')
    parser.add_argument('--bis_npz_dir', type=str, default='./preparation/imf_bis_data/', help='IMF bispectrum matrix directory as .npz')
    parser.add_argument('--test_size', type=float, default=0.3, help='the number of testing dataset')
    parser.add_argument('--val_size', type=float, default=0.1, help='the number of validate dataset')
    parser.add_argument('--random_state', type=int, default=42, help='seed for random initialisation')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay rate')
    parser.add_argument("--batch_size", type=int, default=20, help="Number of batch size.")
    parser.add_argument('--episode_num', type=int, default=5, help='number for training iteration')
    args = parser.parse_args()

    classes = ['W', 'N1', 'N2', 'N3', 'REM']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare_imf_bis(args)
    x_train, y_train, x_val, y_val, x_test, y_test = list_loader(args)
    print("y_train:", Counter(y_train))
    print("y_test:", Counter(y_test))
    train_dataset = MyDataset(x_train, y_train)
    val_dataset = MyDataset(x_val, y_val)
    test_dataset = MyDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # print("train.size=", len(train_loader))
    # print("test.size=", len(test_loader))

    # init model
    # model = LeNet().to(device)
    # model = AlexNet(num_classes=len(classes)).to(device)
    # model = DenseNet(growthRate=4, depth=16, reduction=0.5, nClasses=5, bottleneck=1).to(device)
    model = SEDenseNet(num_init_features=64, growth_rate=4, block_config=(6, 12, 24, 16), bn_size=4, drop_rate=0.8, num_classes=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(num_class=n_classes, alpha=0.7, gamma=5.0, balance_index=2)
    criterion.to(device)

    print("=====Training Phase=====")
    # total_loss = 0.0
    loss_list = list()
    # length = 0
    torch.cuda.empty_cache()
    for episode in range(args.episode_num):
        print("===Episode {}/{}===".format(episode+1, args.episode_num))
        model.zero_grad()
        for step, (batch_x, batch_y) in enumerate(train_loader):
            print('Episode:', episode+1, '| Step:', step+1, '| batch x:', batch_x.size(), '| batch y:', batch_y.size())
            optimizer.zero_grad()
            pred_y = model(batch_x.to(device))
            # loss = criterion(predict, target)
            loss = criterion(pred_y, batch_y.long().to(device))
            loss.backward()
            optimizer.step()
            # total_loss += loss.item()
            loss_list.append(loss.item()/args.batch_size)
            train_acc = accuracy_score(batch_y.cpu().numpy(), torch.max(pred_y, 1)[1].cpu().numpy())

            model.eval()
            val_acc = np.zeros(len(val_loader))
            for val_step, (batch_val, target_val) in enumerate(val_loader):
                pred_val = model(batch_val.to(device))
                y_val_pred = torch.max(pred_val, 1)[1].cpu().numpy()
                # val_acc[val_step] = accuracy_score(target_val.cpu().numpy(), torch.max(y_val_pred, 1)[1].cpu().numpy())
                val_acc[val_step] = accuracy_score(target_val.cpu().numpy(), y_val_pred)
            print("\tLoss=", loss.item(), "| Train Accuracy=", train_acc, "| Val Accuracy={:.3f}".format(np.mean(val_acc)))
            cm = confusion_matrix(y_val, y_val_pred, labels=range(len(classes)))
            ck_score = cohen_kappa_score(y_val, y_val_pred)
            acc_avg, acc, f1_macro, f1, sensitivity, specificity, precision = evaluate_metrics(cm)
            print('Average Accuracy -> {:>6.4f}, Macro F1 -> {:>6.4f} and Cohen\'s Kappa -> {:>6.4f} on test set'.format(acc_avg, f1_macro, ck_score))
            for index_ in range(len(classes)):
                print("\t{} rhythm -> Sensitivity: {:1.4f}, Specificity: {:1.4f}, Precision: {:1.4f}, F1 : {:1.4f} Accuracy: {:1.4f}".format(
                        classes[index_], sensitivity[index_], specificity[index_], precision[index_], f1[index_], acc[index_]))
            print("\tAverage -> Sensitivity: {:1.4f}, Specificity: {:1.4f}, Precision: {:1.4f}, F1-score: {:1.4f}, Accuracy: {:1.4f}".format(
                    np.mean(sensitivity), np.mean(specificity), np.mean(precision), np.mean(f1), np.mean(acc)))

    torch.save(model, 'mySEDenseNet.pth')  # save net model and parameters

    print("=====Testing Phase=====")
    model.eval()
    # for step, (batch_x, batch_y) in enumerate(test_loader):
    #
    # cm = confusion_matrix(y_test, y_pred, labels=range(len(classes)))
    # ck_score = cohen_kappa_score(y_test, y_pred)
    # acc_avg, acc, f1_macro, f1, sensitivity, specificity, precision = evaluate_metrics(cm)
    # print('Average Accuracy -> {:>6.4f}, Macro F1 -> {:>6.4f} and Cohen\'s Kappa -> {:>6.4f} on test set'.format(acc_avg, f1_macro, ck_score))
    # for index_ in range(len(classes)):
    #     print("\t{} rhythm -> Sensitivity: {:1.4f}, Specificity: {:1.4f}, Precision: {:1.4f}, F1 : {:1.4f} Accuracy: {:1.4f}".format(
    #             classes[index_], sensitivity[index_], specificity[index_], precision[index_], f1[index_], acc[index_]))
    # print("\tAverage -> Sensitivity: {:1.4f}, Specificity: {:1.4f}, Precision: {:1.4f}, F1-score: {:1.4f}, Accuracy: {:1.4f}".format(
    #         np.mean(sensitivity), np.mean(specificity), np.mean(precision), np.mean(f1), np.mean(acc)))


if __name__ == "__main__":
    main_ml()
