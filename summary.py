#! /usr/bin/python
# -*- coding: utf8 -*-

import argparse
import os
import re
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix, f1_score

classes = ['W', 'N1', 'N2', 'N3', 'REM']

font = {'family': 'SimHei'}
mpl.rc('font', **font)  # 一次定义终身使用

def evaluate_metrics(cm):
    print("Confusion Matrix:")
    print(cm)
    draw_confusion_matrix(cm)

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

    print("===Test Samples===")
    print("\t total: {}".format(int(np.sum(cm))))
    for index_ in range(len(classes)):
        print("\t {}: {}".format(classes[index_], int(TP[index_] + FN[index_])))

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    # ACC_micro = (sum(TP) + sum(TN)) / (sum(TP) + sum(FP) + sum(FN) + sum(TN))
    ACC_macro = np.mean(ACC) # to get a sense of effectiveness of our method on the small classes we computed this average (macro-average)

    F1 = (2 * PPV * TPR) / (PPV + TPR)
    F1_macro = np.mean(F1)

    return ACC_macro, ACC, F1_macro, F1, TPR, TNR, PPV


def draw_confusion_matrix(cm):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap('cool'))  # 按照像素显示出矩阵
    # plt.title('Confusion Matrix')
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    thresh = cm.max() / 2.
    # iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
    real = np.zeros(5)
    # ij配对，遍历矩阵迭代器
    iters = np.reshape([[[i, j] for j in range(len(classes))] for i in range(len(classes))], (cm.size, 2))
    for i, j in iters:
        real[i] += cm[i, j]
    for i, j in iters:
        plt.text(j, i, format(cm[i, j]/real[i], '.2f'), fontsize=24, ha='center')  # 显示对应的比例
        # plt.text(j, i, format(cm[i, j]), fontsize=24, ha='center')  # 显示对应的数量

    # plt.ylabel('Real Label', fontsize=20)
    # plt.xlabel('Predict Label', fontsize=20)

    plt.ylabel('实际', fontsize=24)
    plt.xlabel('预测', fontsize=24)
    plt.tight_layout()
    plt.show()


def print_performance(cm, y_true, y_pred):
    tp = np.diagonal(cm).astype(np.float)
    tpfp = np.sum(cm, axis=0).astype(np.float) # sum of each col
    tpfn = np.sum(cm, axis=1).astype(np.float) # sum of each row
    acc = np.sum(tp)/ np.sum(cm)
    precision = tp / tpfp
    recall = tp / tpfn
    f1 = (2 * precision * recall) / (precision + recall)

    FP = cm.sum(axis=0).astype(np.float) - np.diag(cm)
    FN = cm.sum(axis=1).astype(np.float) - np.diag(cm)
    TP = np.diag(cm).astype(np.float)
    TN = cm.sum().astype(np.float) - (FP + FN + TP)
    specificity = TN / (TN + FP) #TNR

    mf1 = np.mean(f1)

    print ("Sample: {}".format(np.sum(cm)))
    print ("W: {}".format(tpfn[W]))
    print ("N1: {}".format(tpfn[N1]))
    print ("N2: {}".format(tpfn[N2]))
    print ("N3: {}".format(tpfn[N3]))
    print ("REM: {}".format(tpfn[REM]))
    print ("Confusion matrix:")
    print (cm)
    print ("Precision(PPV): {}".format(precision))
    print ("Recall(Sensitivity): {}".format(recall))
    print ("Specificity: {}".format(specificity))
    print ("F1: {}".format(f1))
    if (len(y_true)>0):
       print ("Overall accuracy: {}".format(np.mean(y_true == y_pred)))
       print ("Cohen's kappa score: {}".format(cohen_kappa_score(y_true, y_pred)))

    else:
        print ("Overall accuracy: {}".format(acc))
    print ("Macro-F1 accuracy: {}".format(mf1))


def perf_overall(data_dir):
    # Remove non-output files, and perform ascending sort
    allfiles = os.listdir(data_dir)
    outputfiles = []
    for idx, f in enumerate(allfiles):
        if re.match("^output_.+\d+\.npz", f):
            outputfiles.append(os.path.join(data_dir, f))
    outputfiles.sort()

    y_true = []
    y_pred = []
    for fpath in outputfiles:
        with np.load(fpath) as f:
            print(f["y_true"].shape)
            if len(f["y_true"].shape) == 1:
                if len(f["y_true"]) < 10:
                    f_y_true = np.hstack(f["y_true"])
                    f_y_pred = np.hstack(f["y_pred"])
                else:
                    f_y_true = f["y_true"]
                    f_y_pred = f["y_pred"]
            else:
                f_y_true = f["y_true"].flatten()
                f_y_pred = f["y_pred"].flatten()

            y_true.extend(f_y_true)
            y_pred.extend(f_y_pred)

            print ("File: {}".format(fpath))
            cm = confusion_matrix(f_y_true, f_y_pred, labels=[0, 1, 2, 3, 4])
            print_performance(cm)
    print(" ")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sio.savemat('con_matrix_sleep.mat',{'y_true': y_true, 'y_pred': y_pred})
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    acc = np.mean(y_true == y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")

    total = np.sum(cm, axis=1)

    print("Ours:")
    print_performance(cm, y_true, y_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="outputs_2013/outputs_eeg_fpz_cz",
                        help="Directory where to load training data_2013.")
    args = parser.parse_args()

    if args.data_dir is not None:
        perf_overall(data_dir=args.data_dir)


if __name__ == "__main__":
    main()
