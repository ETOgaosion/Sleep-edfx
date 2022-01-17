# coding=utf-8
# run with RNN models and mahalanobis distance attention
# 18 feature parameters
# writer: Owens

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy import io
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from models.model_LSTM import *
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from summary import evaluate_metrics
from torch.optim.lr_scheduler import StepLR


def data_loader(args):
    print("===loading feature matrix===")
    x = []
    y = []
    for mat_idx in os.listdir(args.fea_dir):
        mat_name = args.fea_dir + "/" + mat_idx
        fea_loader = io.loadmat(mat_name)
        x.append(fea_loader['fea'])
        y.append(fea_loader['labels'].squeeze())

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=args.test_size, random_state=args.random_state)

    return x_train, y_train, x_test, y_test


# Divide training samples into different pools by stages
def build_pool(x_train, y_train):
    print("===random sample and build pools===")
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    rus = RandomUnderSampler(random_state=0)
    x_train, y_train = rus.fit_sample(x_train, y_train)

    pools = [[] for i in range(5)]
    for epoch_idx in range(len(x_train)):
        pools[y_train[epoch_idx]].append(x_train[epoch_idx])

    print("W stage pool:", np.array(pools[0]).shape)
    print("N1 stage pool:", np.array(pools[1]).shape)
    print("N2 stage pool:", np.array(pools[2]).shape)
    print("N3 stage pool:", np.array(pools[3]).shape)
    print("REM stage pool:", np.array(pools[4]).shape)

    mean = [[] for i in range(5)]
    cov_inv = [[] for i in range(5)]
    for stage_idx in range(len(pools)):
        mean[stage_idx].append(np.mean(np.array(pools[stage_idx]), axis=0))  # mean of pool samples
        cov_inv[stage_idx].append(np.linalg.inv(np.cov(np.array(pools[stage_idx]).T)))  # the inverse of the covariance matrix

    return mean, cov_inv


# Calculate the Mahalanobis distance from the sample to each pool
def cal_mah_distance(epoch, mean, cov_inv, classes):
    d_mah = np.zeros(len(classes))
    for stage_idx in range(len(classes)):
        delta = epoch - mean[stage_idx]
        d_mah[stage_idx] = np.sqrt(np.dot(np.dot(delta, cov_inv[stage_idx]), delta.T))  # Mahalanobis distance

    return d_mah


def cal_trans_mat(seq_target, classes, father):
    seq_target = seq_target.numpy()
    trans_mat = np.ones([len(classes), len(classes)])  # init status transition matrix
    for epoch_idx in range(1, len(seq_target)):
        trans_mat[seq_target[epoch_idx-1], seq_target[epoch_idx]] += 1

    if father is None:
        father = 0

    trans_mat[father] /= np.sum(trans_mat[father])

    return Variable(torch.tensor(trans_mat[father])).float()


def cal_beta_score(x_train, mean, cov_inv, classes):
    beta_score = [[] for i in range(len(x_train))]
    for seq_idx in range(len(x_train)):
        seq_input = x_train[seq_idx]
        for time_step in range(len(seq_input)):
            d_mah = cal_mah_distance(seq_input[time_step], mean, cov_inv, classes)
            beta = np.exp(-d_mah) / sum(np.exp(-d_mah))
            beta_score[seq_idx].append(beta)
    print("====calculate alpha completed===")
    return beta_score


def train_model(model, optimizer, x_train, y_train, episode_num, mean, cov_inv, classes):
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()

    total_loss = 0.0
    loss_list = list()
    length = 0

    # scheduler = StepLR(optimizer, step_size=len(x_train), gamma=0.5)
    print("===Training Phase===")
    for episode in range(episode_num):
        model.zero_grad()
        print("===Episode {}/{}===".format(episode+1, episode_num))
        for seq_idx in range(len(x_train)):
            print("\tTraining Sequence {}: {}".format(seq_idx+1, x_train[seq_idx].shape))
            optimizer.zero_grad()
            seq_input = Variable(torch.tensor(x_train[seq_idx]), requires_grad=False).float()  # [T, 18]
            seq_target = Variable(torch.tensor(y_train[seq_idx]), requires_grad=False).int()  # [T]
            hidden = None  # init hidden 0
            father = None  # record the label of the previous node
            y_pred = torch.tensor([])
            flag = 1  # previous epoch stage right
            for time_step in range(len(seq_input)):
                # print("\ttime_step:{}/{}".format(time_step+1, len(seq_input)))
                d_mah = cal_mah_distance(x_train[seq_idx][time_step], mean, cov_inv, classes)
                beta = Variable(torch.tensor(np.exp(-d_mah) / sum(np.exp(-d_mah))), requires_grad=False).float()
                # gamma = cal_trans_mat(seq_target[:time_step], classes, father)
                # pred_t, hidden = model(seq_input[time_step], beta, gamma, flag, time_step, hidden, father)
                pred_t, hidden = model(seq_input[time_step], beta, time_step, hidden, father)
                y_pred = torch.cat([y_pred, pred_t])
                if seq_target[time_step].item() != torch.max(pred_t, 1)[1].item():
                    flag = np.abs(flag-1)  # previous epoch stage wrong
                # if time_step % (42+episode) == 0:
                #     flag = np.abs(flag - 1)  # random update flag
                father = torch.tensor([seq_target[time_step]])
            loss = criterion(y_pred, seq_target.long())
            loss.backward()
            optimizer.step()
            # scheduler.step()
            length += len(seq_input)
            total_loss += loss.item()
            loss_list.append(total_loss/length)
            acc = accuracy_score(seq_target.numpy(), torch.max(y_pred, 1)[1].numpy())
            print("\tTotal Loss=", total_loss/length)
            print("\tAccuracy=", acc)

    # draw loss figure
    plt.figure()
    plt.plot(loss_list)
    plt.show()

    return model


def evaluate_test_set(model, x_test, y_test, mean, cov_inv, classes):
    print("===Testing Phase===")
    model.eval()
    y_true = list()
    y_pred = list()

    for seq_idx in range(len(x_test)):
        print("\tTesting Sequence {}: {}".format(seq_idx+1, x_test[seq_idx].shape))
        seq_input = Variable(torch.tensor(x_test[seq_idx]), requires_grad=False).float()  # [T, 18]
        seq_target = Variable(torch.tensor(y_test[seq_idx]), requires_grad=False).int()  # [T]
        hidden = None
        father = None
        predict = torch.tensor([])
        flag = 1  # previous epoch stage right
        for time_step in range(len(seq_input)):
            # print("\ttime_step:{}/{}".format(time_step+1, len(seq_input)))
            d_mah = cal_mah_distance(x_test[seq_idx][time_step], mean, cov_inv, classes)
            beta = Variable(torch.tensor(np.exp(-d_mah) / sum(np.exp(-d_mah))), requires_grad=False).float()
            # gamma = cal_trans_mat(seq_target[:time_step], classes, father)
            # pred_t, hidden = model(seq_input[time_step], beta, gamma, flag, time_step, hidden, father)
            pred_t, hidden = model(seq_input[time_step], beta, time_step, hidden, father)
            predict = torch.cat([predict, pred_t])
            if seq_target[time_step].item() != torch.max(pred_t, 1)[1].item():
                flag = np.abs(flag-1)  # previous epoch stage wrong
                # print("label={}, pred={}, beta={}, father={}".format(classes[seq_target[time_step].item()], classes[torch.max(pred_t, 1)[1].item()], beta, father))
            # if time_step % 42 == 0:
            #     flag = np.abs(flag - 1)  # random update flag
            father = torch.tensor([seq_target[time_step]])

        y_true += list(np.array(seq_target))
        y_pred += list(np.array(torch.max(predict, 1)[1]))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    ck_score = cohen_kappa_score(y_true, y_pred)
    acc_avg, acc, f1_macro, f1, sensitivity, specificity, precision = evaluate_metrics(cm)
    print('Average Accuracy -> {:>6.4f}, Macro F1 -> {:>6.4f} and Cohen\'s Kappa -> {:>6.4f} on test set'.format(acc_avg, f1_macro, ck_score))
    for index_ in range(len(classes)):
        print("\t{} rhythm -> Sensitivity: {:1.4f}, Specificity: {:1.4f}, Precision: {:1.4f}, F1-score: {:1.4f} Accuracy: {:1.4f}".format(
                classes[index_], sensitivity[index_], specificity[index_], precision[index_], f1[index_], acc[index_]))
    print("\tAverage -> Sensitivity: {:1.4f}, Specificity: {:1.4f}, Precision: {:1.4f}, F1-score: {:1.4f}, Accuracy: {:1.4f}".format(
            np.mean(sensitivity), np.mean(specificity), np.mean(precision), np.mean(f1), np.mean(acc)))


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--fea_dir', type=str, default='./preparation/fea_par/', help='feature matrix directory')
    parser.add_argument('--fea_dir', type=str, default='./preparation/subbands_fea/', help='feature matrix directory')
    parser.add_argument('--random_state', type=int, default=42, help='seed for random initialisation')
    parser.add_argument('--episode_num', type=int, default=5, help='number for training iteration')
    parser.add_argument('--test_size', type=float, default=0.2, help='the number of testing dataset')
    parser.add_argument('--input_size', '-is', type=int, default=18, help='feature vector size')
    parser.add_argument('--seq_hidden_size', '-hs', type=int, default=128, help='sequence embedding size')
    parser.add_argument('--output_size', '-os', type=int, default=5, help='label class size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay rate')
    parser.add_argument('-k', type=int, default=10, help='use top k similar features to predict')
    parser.add_argument('-w', '--with_last', action='store_true', help='with last h')
    parser.add_argument('-exp_decay', type=float, default=0.9, help='exponential decay learning rate')
    parser.add_argument('-beta_threshold', type=float, default=0.55, help='the threshold of beta score')

    args = parser.parse_args()

    classes = ['W', 'N1', 'N2', 'N3', 'REM']

    x_train, y_train, x_test, y_test = data_loader(args)
    mean, cov_inv = build_pool(x_train, y_train)
    # beta_score = cal_beta_score(x_train, mean, cov_inv, classes)

    # model = LSTMM(args)
    model = LSTMMAD(args)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model = train_model(model, optimizer, x_train, y_train, args.episode_num, mean, cov_inv, classes)
    evaluate_test_set(model, x_test, y_test, mean, cov_inv, classes)


if __name__ == "__main__":
    main()
