# coding=utf-8
# run with classical machine learning methods
# training on SleepEDF dataset, testing on PKU private dataset
# 18 feature parameters
# writer: Owens

import os
import argparse
import numpy as np
from scipy import io
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import EasyEnsemble
from collections import Counter
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from summary import evaluate_metrics


def loader(args):
    print("===loading training set===")
    x_train = []
    y_train = []
    for mat_idx in os.listdir(args.train_dir):
        mat_name = args.train_dir + "/" + mat_idx
        fea_loader = io.loadmat(mat_name)
        x_train.append(fea_loader['fea'])
        y_train.append(fea_loader['labels'].squeeze())
        print("fea:", fea_loader['fea'].shape, "labels:", fea_loader['labels'].shape)
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    # y_train = np.array([np.argmax(label) for label in y_train])
    print("===loading testing set===")
    x_test = []
    y_test = []
    for mat_idx in os.listdir(args.test_dir):
        mat_name = args.test_dir + "/" + mat_idx
        fea_loader = io.loadmat(mat_name)
        x_test.append(fea_loader['fea'])
        y_test.append(fea_loader['labels'].squeeze())
        print("fea:", fea_loader['fea'].shape, "labels:", fea_loader['labels'].shape)
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)

    return x_train, y_train, x_test, y_test


def main_ml():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='./preparation/subbands_fea/', help='feature matrix directory of training set')
    parser.add_argument('--test_dir', type=str, default='./preparation/pku_fea/', help='feature matrix directory of testing set')
    parser.add_argument('--test_size', type=float, default=0.3, help='the number of testing dataset')
    parser.add_argument('--random_state', type=int, default=42, help='seed for random initialisation')
    args = parser.parse_args()

    classes = ['W', 'N1', 'N2', 'N3', 'REM']

    x_train, y_train, x_test, y_test = loader(args)
    print("y_train:", (Counter(y_train)))
    print("y_test:", Counter(y_test))

    # Resampling
    # ada = ADASYN(random_state=args.random_state)
    # x_train, y_train = ada.fit_sample(x_train, y_train)
    # print("after over-sampling: y_train=", Counter(y_train))
    # rus = RandomUnderSampler(random_state=args.random_state)
    # x_train, y_train = rus.fit_sample(x_train, y_train)
    # print("after under-sampling: y_train=", Counter(y_train))

    print("===Training Phase===")

    # SVM Classifier:
    # clf = svm.SVC(C=1.0, kernel='rbf', gamma='auto', class_weight='balanced', decision_function_shape='ovr')
    # clf.fit(x_train, y_train)
    # print(clf.score(x_train, y_train))
    # y_pred = clf.predict(x_test)

    # kNN Classifier:
    # knn = KNeighborsClassifier()
    # knn.fit(x_train, y_train)
    # print(knn.score(x_train, y_train))  # 精度
    # y_pred = knn.predict(x_test)

    # Bayes Classifier:
    # gnb = GaussianNB()
    # gnb.fit(x_train, y_train)
    # print(gnb.score(x_train, y_train))  # 精度
    # y_pred = gnb.predict(x_test)

    # Decision Tree Classifier:
    # dtc = DecisionTreeClassifier(max_leaf_nodes=10)
    # dtc.fit(x_train, y_train)
    # print(dtc.score(x_train, y_train))  # 精度
    # y_pred = dtc.predict(x_test)

    # CatBoost Classifier:
    # model = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.5, loss_function='MultiClass')
    # model.fit(x_train, y_train)
    # print("Training Accuracy=", model.score(x_train, y_train))
    # y_pred = model.predict(x_test)

    # XgBoost Classifier:
    model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100, silent=True, objective='multi:softmax')
    model.fit(x_train, y_train)
    print("Training Accuracy=", model.score(x_train, y_train))
    y_pred = model.predict(x_test)

    xgb.plot_importance(model)
    plt.show()
    digraph = xgb.to_graphviz(model, num_trees=1)
    digraph.format = 'png'
    digraph.view('./images/xgb_tree')

    print("===Testing Phase===")
    cm = confusion_matrix(y_test, y_pred, labels=range(len(classes)))
    ck_score = cohen_kappa_score(y_test, y_pred)
    acc_avg, acc, f1_macro, f1, sensitivity, specificity, precision = evaluate_metrics(cm)
    print(
        'Average Accuracy -> {:>6.4f}, Macro F1 -> {:>6.4f} and Cohen\'s Kappa -> {:>6.4f} on test set'.format(acc_avg,
                                                                                                               f1_macro,
                                                                                                               ck_score))
    for index_ in range(len(classes)):
        print(
            "\t{} rhythm -> Sensitivity: {:1.4f}, Specificity: {:1.4f}, Precision: {:1.4f}, F1 : {:1.4f} Accuracy: {:1.4f}".format(
                classes[index_], sensitivity[index_], specificity[index_], precision[index_], f1[index_], acc[index_]))
    print(
        "\tAverage -> Sensitivity: {:1.4f}, Specificity: {:1.4f}, Precision: {:1.4f}, F1-score: {:1.4f}, Accuracy: {:1.4f}".format(
            np.mean(sensitivity), np.mean(specificity), np.mean(precision), np.mean(f1), np.mean(acc)))


if __name__ == "__main__":
    main_ml()