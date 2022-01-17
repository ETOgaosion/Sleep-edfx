# coding=utf-8
# run with classical machine learning methods
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
# from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from summary import evaluate_metrics


def loader(args):
    print("===loading feature matrix===")
    x = []
    y = []
    for mat_idx in os.listdir(args.fea_dir):
        mat_name = args.fea_dir + "/" + mat_idx
        fea_loader = io.loadmat(mat_name)
        # temp = [fea_loader['fea']]
        x.append(fea_loader['fea'])
        # y.append(fea_loader['labels'])
        y.append(fea_loader['labels'].squeeze())
        # print("fea:", fea_loader['fea'].shape, "labels:", fea_loader['labels'].shape)
        # print("fea:", temp[:, :-5].shape, "labels:", fea_loader['labels'].shape)

    x = np.concatenate(x)
    y = np.concatenate(y)
    # y = np.array([np.argmax(label) for label in y])

    print("y:", Counter(y), "total:", y.size)
    # ada = ADASYN(random_state=args.random_state)
    # x, y = ada.fit_sample(x, y)
    # print("after over-sampling: y=", Counter(y))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=args.test_size, random_state=args.random_state)
    return x_train, y_train, x_test, y_test


def main_ml():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--fea_dir', type=str, default='./preparation/fea_new/', help='feature matrix directory')
    # parser.add_argument('--fea_dir', type=str, default='./preparation/subbands_fea/', help='feature matrix directory')
    # parser.add_argument('--fea_dir', type=str, default='./preparation/fea_par/', help='feature matrix directory')
    parser.add_argument('--fea_dir', type=str, default='./preparation/pku_fea/', help='feature matrix directory')
    # parser.add_argument('--fea_dir', type=str, default='./preparation/mmd_fea/', help='feature matrix directory')
    parser.add_argument('--test_size', type=float, default=0.3, help='the number of testing dataset')
    parser.add_argument('--random_state', type=int, default=42, help='seed for random initialisation')
    args = parser.parse_args()

    classes = ['W', 'N1', 'N2', 'N3', 'REM']

    x_train, y_train, x_test, y_test = loader(args)

    print("y_train:", Counter(y_train))
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
    # print(model.score(x_train, y_train))
    # y_pred = model.predict(x_test)

    # XgBoost Classifier:
    model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100, silent=True, objective='multi:softmax')
    model.fit(x_train, y_train)
    print(model.score(x_train, y_train))
    y_pred = model.predict(x_test)
    xgb.plot_importance(model)
    plt.show()
    # digraph = xgb.to_graphviz(model, num_trees=1)
    # digraph.format = 'png'
    # digraph.view('./images/xgb_tree')

    print("===Testing Phase===")
    cm = confusion_matrix(y_test, y_pred, labels=range(len(classes)))
    ck_score = cohen_kappa_score(y_test, y_pred)
    acc_avg, acc, f1_macro, f1, sensitivity, specificity, precision = evaluate_metrics(cm)
    print('Average Accuracy -> {:>6.4f}, Macro F1 -> {:>6.4f} and Cohen\'s Kappa -> {:>6.4f} on test set'.format(acc_avg, f1_macro, ck_score))
    for index_ in range(len(classes)):
        print("\t{} rhythm -> Sensitivity: {:1.4f}, Specificity: {:1.4f}, Precision: {:1.4f}, F1 : {:1.4f} Accuracy: {:1.4f}".format(
                classes[index_], sensitivity[index_], specificity[index_], precision[index_], f1[index_], acc[index_]))
    print("\tAverage -> Sensitivity: {:1.4f}, Specificity: {:1.4f}, Precision: {:1.4f}, F1-score: {:1.4f}, Accuracy: {:1.4f}".format(
            np.mean(sensitivity), np.mean(specificity), np.mean(precision), np.mean(f1), np.mean(acc)))


if __name__ == "__main__":
    main_ml()
