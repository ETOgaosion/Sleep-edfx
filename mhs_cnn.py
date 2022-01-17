#! /usr/bin/python
# -*- coding: utf8 -*-

import argparse
import os

import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score

from dataloader import SeqDataLoader
from models import model_MHS
from models import focal_loss

# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4

classes = ['W', 'N1', 'N2', 'N3', 'REM']
n_classes = len(classes)


def evaluate_metrics(cm, classes):
    print ("Confusion matrix:")
    print (cm)
    img = plt.imshow(cm, cmap=plt.cm.winter)
    plt.show(img)

    cm = cm.astype(np.float32)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    # https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    # ACC_micro = (sum(TP) + sum(TN)) / (sum(TP) + sum(FP) + sum(FN) + sum(TN))
    ACC_macro = np.mean(ACC) # to get a sense of effectiveness of our method on the small classes we computed this average (macro-average)

    F1 = (2 * PPV * TPR) / (PPV + TPR)
    F1_macro = np.mean(F1)

    print ("Sample: {}".format(int(np.sum(cm))))

    for index_ in range(n_classes):
        print ("{}: {}".format(classes[index_], int(TP[index_] + FN[index_])))

    return ACC_macro, ACC, F1_macro, F1, TPR, TNR, PPV


def evaluate_model(y_true, y_pred, classes):
    n_classes = len(classes)
    alignments_alphas_all = []

    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    ck_score = cohen_kappa_score(y_true, y_pred)
    acc_avg, acc, f1_macro, f1, sensitivity, specificity, PPV = evaluate_metrics(cm, classes)
    print(
        'Average Accuracy -> {:>6.4f}, Macro F1 -> {:>6.4f} and Cohen\'s Kappa -> {:>6.4f} on test set'.format(acc_avg,
                                                                                                               f1_macro,
                                                                                                               ck_score))
    for index_ in range(n_classes):
        print(
            "\t{} rhythm -> Sensitivity: {:1.4f}, Specificity: {:1.4f}, Precision (PPV): {:1.4f}, F1 : {:1.4f} Accuracy: {:1.4f}".format(
                classes[index_],
                sensitivity[
                    index_],
                specificity[
                    index_], PPV[index_], f1[index_],
                acc[index_]))
    print(
        "\tAverage -> Sensitivity: {:1.4f}, Specificity: {:1.4f}, Precision (PPV): {:1.4f}, F1-score: {:1.4f}, Accuracy: {:1.4f}".format(
            np.mean(sensitivity), np.mean(specificity), np.mean(PPV), np.mean(f1), np.mean(acc)))

    return acc_avg, f1_macro, ck_score, y_true, y_pred, alignments_alphas_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data_2013/eeg_fpz_cz",
                        help="Directory where to load prediction outputs")
    parser.add_argument("--mhs_dir", type=str, default="data_2013/mhs",
                        help="Directory where to load mhs outputs")
    parser.add_argument("--train_dir", type=str, default="data_2013/mhs_train_data",
                        help="Directory where to store train_data after sampling")
    parser.add_argument("--output_dir", type=str, default="outputs_2013/outputs_mhs_fpz_cz",
                        help="Directory where to save trained models and outputs")
    parser.add_argument("--num_folds", type=int, default=10,
                        help="Number of cross-validation folds.")
    parser.add_argument("--batch_size", type=int, default=200,
                        help="Number of batch size.")
    parser.add_argument("--mhs_num", type=int, default=8,
                        help="Number of selected mhs.")
    parser.add_argument("--checkpoint_dir", type=str, default="ckpt_2013/mhs", help="Directory to save checkpoints")
    args = parser.parse_args()

    char2numY = dict(zip(classes, range(len(classes))))
    pre_f1_macro = 0

    # gpu or not
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(str(datetime.now()))
    for fold_idx in range(args.num_folds):
    #for fold_idx in range(1):
        start_time_fold_i = time.time()
        print("fold_idx={}".format(fold_idx))
        data_loader = SeqDataLoader(args.mhs_dir, args.num_folds, fold_idx, classes=classes)
        X_train, y_train, X_test, y_test = data_loader.load_data(seq_len=args.batch_size)
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        # take 2000 of every epoch in training set
        under_sample_len = 5000
        X_train = np.reshape(X_train, [X_train.shape[0] * X_train.shape[1], -1])
        y_train = y_train.flatten()

        for i in range(n_classes):
            Epochs = np.where(y_train == i)[0]
            len_E = len(np.where(y_train == i)[0])
            len_r = len_E - under_sample_len if (len_E - under_sample_len) > 0 else 0
            permute = np.random.permutation(len_E)
            permute = permute[:len_r]
            y_train = np.delete(y_train, Epochs[permute], axis=0)
            X_train = np.delete(X_train, Epochs[permute], axis=0)

        X_train = X_train[:(X_train.shape[0] // args.batch_size) * args.batch_size, :]
        y_train = y_train[:(X_train.shape[0] // args.batch_size) * args.batch_size]

        X_train = np.reshape(X_train, [-1, X_test.shape[1], X_test.shape[2], X_test.shape[3]])
        y_train = np.reshape(y_train, [-1, y_test.shape[1], ])

        # shuffle training data_2013
        permute = np.random.permutation(len(y_train))
        X_train = np.asarray(X_train)
        X_train = X_train[permute]
        y_train = y_train[permute]

        y_train = np.array(y_train)
        y_test = np.array(y_test)

        print ('The training set after under sampling: '"", classes)
        for cl in classes:
            print (cl, len(np.where(y_train == char2numY[cl])[0]))

        if (os.path.exists(args.checkpoint_dir) == False):
            os.mkdir(args.checkpoint_dir)

        if (os.path.exists(args.output_dir) == False):
            os.makedirs(args.output_dir)

        # X_train = X_train[:, :, :, 3:]
        # X_test = X_test[:, :, :, 3:]

        # initialize MHSNet, set cross entropy as the loss function, set Adam as the optimizer
        net = MHSNet().to(device)
        criterion = FocalLoss(num_class=n_classes, alpha=0.7, gamma=5.0, balance_index=2)
        optimizer = optim.Adam(net.parameters())

        # Training Phase
        print("=====Training Phase=====")
        running_loss = []
        cur_loss = 0.0

        for batch_i in range(len(X_train)):
            inputs = torch.from_numpy(X_train[batch_i]).float()
            labels = torch.IntTensor(np.array([y_train[batch_i]]).T).long()

            # wrap them in Variable
            inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, torch.squeeze(labels))
            loss.backward()

            optimizer.step()
            cur_loss += loss.item()
            running_loss.append(cur_loss)

        print("Training loss = {}".format(cur_loss))

        plt.plot(running_loss)
        plt.show()

        # Testing Phase
        print("=====Testing Phase=====")
        y_pred = []
        for batch_i in range(len(X_test)):
            inputs = torch.from_numpy(X_test[batch_i]).float()
            predicted = torch.max(net(inputs.to(device)), dim=1)[1].data.cpu().numpy()  # return index of the index value
            y_pred.append(predicted)
        y_pred = np.array(y_pred)

        y_test = y_test.flatten()
        y_pred = y_pred.flatten()

        # evaluate results
        acc_avg, f1_macro, ck_score, y_true, y_pred, alignments_alphas_all = evaluate_model(y_test, y_pred, classes)
        if np.nan_to_num(f1_macro) > pre_f1_macro:  # save the better model based on the f1 score
            pre_f1_macro = f1_macro
            ckpt_name = "model_fold{:02d}.ckpt".format(fold_idx)
            save_path = os.path.join(args.checkpoint_dir, ckpt_name)

            print("The best model (till now) saved in path: %s" % save_path)

            # Save
            save_dict = {
                "y_true": y_true,
                "y_pred": y_pred,
                "ck_score": ck_score,
                "alignments_alphas_all": alignments_alphas_all[:200],
                # we save just the first 200 batch results because it is so huge
            }
            filename = "output_fold{:02d}.npz".format(fold_idx)
            save_path = os.path.join(args.output_dir, filename)
            np.savez(save_path, **save_dict)
            print("The best results (till now) saved in path: %s" % save_path)

        print(str(datetime.now()))
        print ('Fold{} took: {:>6.3f}s'.format(fold_idx, time.time() - start_time_fold_i))


if __name__ == "__main__":
    main()