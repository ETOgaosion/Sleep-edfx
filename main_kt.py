# coding=utf-8
# run with RNN models and experience
# 18 feature parameters
# writer: Owens

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy import io
from scipy.spatial.distance import pdist
from sklearn.model_selection import train_test_split
from models.model_LSTM import *
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from summary import evaluate_metrics
import torch


def loader(args):
    print("===loading feature matrix===")
    x = []
    y = []
    for mat_idx in os.listdir(args.fea_dir):
        mat_name = args.fea_dir + "/" + mat_idx
        fea_loader = io.loadmat(mat_name)
        x.append(fea_loader['fea'])
        y.append(fea_loader['labels'].squeeze())

    # random shuffle
    # random.seed(args.random_state)
    # random.shuffle(x)
    # random.seed(args.random_state)
    # random.shuffle(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=args.test_size, random_state=args.random_state)
    return x_train, y_train, x_test, y_test


# calculate the experience matrix Em
def calculate_experience(x_train, y_train):
    exp_mat = np.zeros((5, 15))
    count_mat = np.zeros(5, dtype=np.int)
    for seq_idx in range(len(x_train)):
        for epoch_idx in range(len(x_train[seq_idx])):
            count_mat[y_train[seq_idx][epoch_idx]] += 1
            exp_mat[y_train[seq_idx][epoch_idx]] += x_train[seq_idx][epoch_idx]
    for stage_idx in range(5):
        exp_mat[stage_idx] = exp_mat[stage_idx]/count_mat[stage_idx]
    print("Experience Matrix:{}".format(exp_mat.shape))
    return exp_mat


# draw the radar maps
def draw_radar_map(values, name, classes):
    # 用于正常显示中文
    plt.rcParams['font.sans-serif'] = 'SimHei'
    # 用于正常显示符号
    plt.rcParams['axes.unicode_minus'] = False
    # 使用ggplot的绘图风格，这个类似于美化了，可以通过plt.style.available查看可选值，你会发现其它的风格真的丑。。。
    plt.style.use('ggplot')

    # 设置每个数据点的显示位置，在雷达图上用角度表示
    angles = np.linspace(0, 2 * np.pi, values.shape[2], endpoint=False)  # [5,3,6]

    # 拼接数据首尾，使图形中线条封闭
    # values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    fig = plt.figure()
    par_type = ['幅值', '能量', '功率']
    for par_idx in range(values.shape[1]):  # [amp2, energy, power]
        for stage_idx in range(values.shape[0]):
            ax = fig.add_subplot(1, 3, par_idx+1, polar=True)
            value = np.concatenate((values[stage_idx, par_idx], [values[stage_idx, par_idx, 0]]))
            ax.plot(angles, value, 'o-', linewidth=2, label=classes[stage_idx])
            ax.fill(angles, value, alpha=0.25)
            ax.set_thetagrids(angles * 180 / np.pi, name)
            plt.title('各分期经验均值脑波{}参数对比'.format(par_type[par_idx]))
            ax.grid(True)
    plt.legend(loc='best')

    plt.show()


def train_model(model, optimizer, x_train, y_train, episode_num):
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()

    total_loss = 0.0
    loss_list = list()
    length = 0

    for episode in range(episode_num):
        print("===Episode {}/{}===".format(episode+1, episode_num))
        for seq_idx in range(len(x_train)):
            print("\tTraining Sequence {}: {}".format(seq_idx+1, x_train[seq_idx].shape))
            model.zero_grad()

            seq_input = Variable(torch.tensor(x_train[seq_idx]), requires_grad=False).float()  # [T, 18]
            seq_target = Variable(torch.tensor(y_train[seq_idx]), requires_grad=False).int()  # [T]
            hidden = None
            y_pred = torch.tensor([])
            for time_step in range(len(seq_input)):
                # print("\ttime_step:{}/{}".format(time_step+1, len(seq_input)))
                pred_t, hidden = model(seq_input[time_step], time_step, hidden)
                y_pred = torch.cat([y_pred, pred_t])
            loss = criterion(y_pred, seq_target.long())
            loss.backward()
            optimizer.step()
            length += len(seq_input)
            total_loss += loss.item()
            loss_list.append(total_loss/length)

            acc = accuracy_score(seq_target.numpy(), torch.max(y_pred, 1)[1].numpy())
            print("\tTotal Loss=", total_loss/length)
            print("\tOverall Accuracy=", acc)

    # draw loss figure
    plt.figure()
    plt.plot(loss_list)
    plt.show()

    return model


def evaluate_test_set(model, x_test, y_test, classes):
    y_true = list()
    y_pred = list()

    for seq_idx in range(len(x_test)):
        print("\tTesting Sequence {}: {}".format(seq_idx+1, x_test[seq_idx].shape))
        seq_input = Variable(torch.tensor(x_test[seq_idx]), requires_grad=False).float()  # [T, 18]
        seq_target = Variable(torch.tensor(y_test[seq_idx]), requires_grad=False).int()  # [T]
        hidden = None
        predict = torch.tensor([])
        for time_step in range(len(seq_input)):
            # print("\ttime_step:{}/{}".format(time_step+1, len(seq_input)))
            pred_t, hidden = model(seq_input[time_step], time_step, hidden)
            predict = torch.cat([predict, pred_t])
        y_true += list(np.array(seq_target))
        y_pred += list(np.array(torch.max(predict, 1)[1]))

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
    parser.add_argument('--fea_dir', type=str, default='./preparation/subbands_fea2/', help='feature matrix directory')
    parser.add_argument('--random_state', type=int, default=42, help='seed for random initialisation')
    parser.add_argument('--episode_num', type=int, default=5, help='number for training iteration')
    parser.add_argument('--test_size', type=float, default=0.2, help='the number of testing dataset')
    parser.add_argument('--input_size', '-is', type=int, default=18, help='feature vector size')
    parser.add_argument('--seq_hidden_size', '-hs', type=int, default=128, help='sequence embedding size')
    parser.add_argument('--output_size', '-os', type=int, default=5, help='label class size')
    parser.add_argument('--score_mode', '-s', choices=['concat', 'double'], default='concat',
                        help='way to combine topics and scores')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay rate')
    args = parser.parse_args()

    classes = ['W', 'N1', 'N2', 'N3', 'REM']
    # fea_name = ['slow_amp', 'delta_amp', 'theta_amp', 'alpha_amp', 'beta_amp', 'gamma_amp',
    #             'slow_energy', 'delta_energy', 'theta_energy', 'alpha_energy', 'beta_energy', 'gamma_energy',
    #             'slow_power', 'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power']
    waves = ['delta', 'theta', 'alpha', 'beta', 'gamma']

    x_train, y_train, x_test, y_test = loader(args)
    exp_mat = calculate_experience(x_train, y_train)  # experience matrix
    exp_mat = np.abs(exp_mat)
    exp_mat = exp_mat.reshape([5, 3, 5])

    draw_radar_map(exp_mat, waves, classes)

    # model = LSTM(args)
    #
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # model = train_model(model, optimizer, x_train, y_train, args.episode_num)
    # evaluate_test_set(model, x_test, y_test, classes)


if __name__ == "__main__":
    main()
