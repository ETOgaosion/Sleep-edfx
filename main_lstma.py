# coding=utf-8
# run with Attention-based RNN models
# 18 feature parameters
# writer: Owens

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy import io
from sklearn.model_selection import train_test_split
from models.model_LSTM import *
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from summary import evaluate_metrics
from torch.optim.lr_scheduler import StepLR

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


def train_model(model, optimizer, x_train, y_train, episode_num):
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()

    total_loss = 0.0
    loss_list = list()
    length = 0

    scheduler = StepLR(optimizer, step_size=len(x_train), gamma=0.5)
    for episode in range(episode_num):
        print("===Episode {}/{}===".format(episode+1, episode_num))
        for seq_idx in range(len(x_train)):
            print("\tTraining Sequence {}: {}".format(seq_idx+1, x_train[seq_idx].shape))
            optimizer.zero_grad()

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
            scheduler.step()
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
        print("\t{} rhythm -> Sensitivity: {:1.4f}, Specificity: {:1.4f}, Precision: {:1.4f}, F1-score: {:1.4f} Accuracy: {:1.4f}".format(
                classes[index_], sensitivity[index_], specificity[index_], precision[index_], f1[index_], acc[index_]))
    print("\tAverage -> Sensitivity: {:1.4f}, Specificity: {:1.4f}, Precision: {:1.4f}, F1-score: {:1.4f}, Accuracy: {:1.4f}".format(
            np.mean(sensitivity), np.mean(specificity), np.mean(precision), np.mean(f1), np.mean(acc)))


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--fea_dir', type=str, default='./preparation/fea_par/', help='feature matrix directory')
    parser.add_argument('--fea_dir', type=str, default='./preparation/subbands_fea/', help='feature matrix directory')
    parser.add_argument('--random_state', type=int, default=42, help='seed for random initialisation')
    parser.add_argument('--episode_num', type=int, default=3, help='number for training iteration')
    parser.add_argument('--test_size', type=float, default=0.2, help='the number of testing dataset')
    parser.add_argument('--input_size', '-is', type=int, default=18, help='feature vector size')
    parser.add_argument('--seq_hidden_size', '-hs', type=int, default=128, help='sequence hidden size')
    parser.add_argument('--output_size', '-os', type=int, default=5, help='label class size')
    parser.add_argument('--score_mode', '-s', choices=['concat', 'double'], default='concat',
                        help='way to combine topics and scores')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay rate')
    parser.add_argument('-k', type=int, default=25, help='use top k similar epochs to predict')
    parser.add_argument('-w', '--with_last', action='store_false', help='with last h')
    parser.add_argument('-exp_decay', type=float, default=0.9, help='exponential decay learning rate')

    args = parser.parse_args()

    classes = ['W', 'N1', 'N2', 'N3', 'REM']

    x_train, y_train, x_test, y_test = loader(args)

    # model = LSTMA(args)
    model = RADecay(args)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model = train_model(model, optimizer, x_train, y_train, args.episode_num)
    evaluate_test_set(model, x_test, y_test, classes)


if __name__ == "__main__":
    main()
