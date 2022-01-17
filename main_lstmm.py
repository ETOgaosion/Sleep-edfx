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
from collections import Counter


def data_loader(args):
    print("===loading feature matrix===")
    x = []
    y = []
    for mat_idx in os.listdir(args.fea_dir):
        mat_name = args.fea_dir + "/" + mat_idx
        fea_loader = io.loadmat(mat_name)
        x.append(fea_loader['fea'])
        y.append(fea_loader['labels'].squeeze())

    print(Counter(y))
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


def train_model(model, device, optimizer, x_train, y_train, episode_num, beta_score):
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()

    # using cuda or cpu
    criterion.to(device)

    total_loss = 0.0
    loss_list = list()
    length = 0

    scheduler = StepLR(optimizer, step_size=len(x_train), gamma=0.5)
    print("===Training Phase===")
    for episode in range(episode_num):
        # model.zero_grad()
        print("===Episode {}/{}===".format(episode+1, episode_num))
        for seq_idx in range(len(x_train)):
            print("\tTraining Sequence {}: {}".format(seq_idx+1, x_train[seq_idx].shape))
            optimizer.zero_grad()
            seq_input = Variable(torch.tensor(x_train[seq_idx]), requires_grad=False).float().to(device)  # [T, 18]
            seq_target = Variable(torch.IntTensor(y_train[seq_idx]), requires_grad=False).long().to(device)  # [T]
            hidden = None  # init hidden 0
            father = None  # record the label of the previous node
            y_pred = torch.tensor([]).to(device)
            for time_step in range(len(seq_input)):
                # print("\ttime_step:{}/{}".format(time_step+1, len(seq_input)))
                beta = Variable(torch.tensor(beta_score[seq_idx][time_step]), requires_grad=False).float().to(device)
                pred_t, hidden = model(seq_input[time_step], beta, time_step, hidden, father)
                y_pred = torch.cat([y_pred, pred_t])
                father = Variable(torch.tensor([seq_target[time_step]])).to(device)
            loss = criterion(y_pred, seq_target)

            loss.backward()
            optimizer.step()
            scheduler.step()

            length += len(seq_input)

            total_loss += loss.item()
            loss_list.append(total_loss/length)
            acc = accuracy_score(seq_target.cpu().numpy(), torch.max(y_pred, 1)[1].cpu().numpy())

            print("\tTotal Loss=", total_loss/length)
            print("\tAccuracy=", acc)

    # draw loss figure
    plt.figure()
    plt.plot(loss_list)
    plt.show()

    return model


def evaluate_test_set(model, device, x_test, y_test, mean, cov_inv, classes):
    print("===Testing Phase===")
    model.eval()
    y_true = list()
    y_pred = list()

    for seq_idx in range(len(x_test)):
        print("\tTesting Sequence {}: {}".format(seq_idx+1, x_test[seq_idx].shape))
        seq_input = Variable(torch.tensor(x_test[seq_idx]), requires_grad=False).float().to(device) # [T, 18]
        seq_target = Variable(torch.IntTensor(y_test[seq_idx]), requires_grad=False).int().to(device)  # [T]
        hidden = None
        father = None
        predict = torch.tensor([]).to(device)
        for time_step in range(len(seq_input)):
            # print("\ttime_step:{}/{}".format(time_step+1, len(seq_input)))
            d_mah = cal_mah_distance(x_test[seq_idx][time_step], mean, cov_inv, classes)
            beta = Variable(torch.tensor(np.exp(-d_mah) / sum(np.exp(-d_mah))), requires_grad=False).float().to(device)
            pred_t, hidden = model(seq_input[time_step], beta, time_step, hidden, father)
            predict = torch.cat([predict, pred_t])

            # if seq_target[time_step].item() != torch.max(pred_t, 1)[1].item():
            #     print("label={}, pred={}, beta={}, time={}, father={}".format(classes[seq_target[time_step].item()],
            #                                                          classes[torch.max(pred_t, 1)[1].item()], beta, time_step, father))
            father = Variable(torch.tensor([seq_target[time_step]])).to(device)

        # fig, ax = plt.subplots()
        # epochs = np.arange(0, len(seq_input), 1)
        # ax.step(epochs, seq_target.cpu().numpy(), 'b-', label='true')
        # ax.step(epochs, torch.max(predict, 1)[1].cpu().numpy(), 'r-', label='predict')
        # plt.yticks(np.arange(5), classes)
        # plt.legend()
        # plt.show()

        y_ture += list(seq_target.cpu().numpy())
        y_pred += list(torch.max(predict, 1)[1].cpu().numpy())

    cm = confusion_matrix(y_ture, y_pred, labels=range(len(classes)))
    ck_score = cohen_kappa_score(y_ture, y_pred)
    acc_avg, acc, f1_macro, f1, sensitivity, specificity, precision = evaluate_metrics(cm)
    print('Average Accuracy -> {:>6.4f}, Macro F1 -> {:>6.4f} and Cohen\'s Kappa -> {:>6.4f} on test set'.format(acc_avg, f1_macro, ck_score))
    for index_ in range(len(classes)):
        print("\t{} rhythm -> Sensitivity: {:1.4f}, Specificity: {:1.4f}, Precision: {:1.4f}, F1-score: {:1.4f} Accuracy: {:1.4f}".format(
                classes[index_], sensitivity[index_], specificity[index_], precision[index_], f1[index_], acc[index_]))
    print("\tAverage -> Sensitivity: {:1.4f}, Specificity: {:1.4f}, Precision: {:1.4f}, F1-score: {:1.4f}, Accuracy: {:1.4f}".format(
            np.mean(sensitivity), np.mean(specificity), np.mean(precision), np.mean(f1), np.mean(acc)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fea_dir', type=str, default='./preparation/fea_new/', help='feature matrix directory')
    # parser.add_argument('--fea_dir', type=str, default='./preparation/subbands_fea/', help='feature matrix directory')
    parser.add_argument('--random_state', type=int, default=42, help='seed for random initialisation')
    parser.add_argument('--episode_num', type=int, default=5, help='number for training iteration')
    parser.add_argument('--test_size', type=float, default=0.2, help='the number of testing dataset')
    parser.add_argument('--input_size', '-is', type=int, default=20, help='feature vector size')
    parser.add_argument('--seq_hidden_size', '-hs', type=int, default=128, help='sequence embedding size')
    parser.add_argument('--output_size', '-os', type=int, default=5, help='label class size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay rate')
    parser.add_argument('-k', type=int, default=25, help='use top k similar features to predict')
    parser.add_argument('-w', '--with_last', action='store_false', help='with last h')
    parser.add_argument('-exp_decay', type=float, default=0.9, help='exponential decay learning rate')
    parser.add_argument('-beta_threshold', type=float, default=0.55, help='the threshold of beta score')

    args = parser.parse_args()

    classes = ['W', 'N1', 'N2', 'N3', 'REM']

    x_train, y_train, x_test, y_test = data_loader(args)
    mean, cov_inv = build_pool(x_train, y_train)
    beta_score = cal_beta_score(x_train, mean, cov_inv, classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = LSTMM(args).to(device)
    model = LSTMMAD(args).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model = train_model(model, device, optimizer, x_train, y_train, args.episode_num, beta_score)
    evaluate_test_set(model, device, x_test, y_test, mean, cov_inv, classes)


if __name__ == "__main__":
    main()
