import numpy as np
import argparse
import net_sphere
import os
from torchtools import cuda, TorchTools, Tensor
from data_loader import get_val_loader

from torch.autograd import Variable

import math


def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = True if float(d[0]) > threshold else False
        y_predict.append(same)
        y_true.append(bool(d[1]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)

    tp = np.sum(np.logical_and(y_predict, y_true))
    fp = np.sum(np.logical_and(y_predict, np.logical_not(y_true)))
    tn = np.sum(np.logical_and(np.logical_not(y_predict), np.logical_not(y_true)))
    fn = np.sum(np.logical_and(np.logical_not(y_predict), y_true))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / len(y_true)

    return tpr, fpr, acc


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        _, _, accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


def KFold(n, n_folds=5, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[int(i * n / n_folds):int(math.ceil((i + 1) * n / n_folds))]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds

def validate(net, data_loader):
    print('Begin validation')
    net.eval()
    dists = []

    for pairs, labels in iter(data_loader):
        # using nets to extract features from pairs and comparing (scoring using cosine distance)
        img_a = Variable(pairs[0]).type(Tensor)
        img_b = Variable(pairs[1]).type(Tensor)
        # img_b = Tensor(pairs[1])

        _, embs_a = net(img_a)
        _, embs_b = net(img_b)

        embs_a = embs_a.data
        embs_b = embs_b.data

        for i in range(len(embs_a)):
            cos_dis = embs_a[i].dot(embs_b[i]) / (embs_a[i].norm() * embs_b[i].norm() + 1e-5)
            dists.append([cos_dis, int(labels[i])])

    dists = np.array(dists)

    tprs = []
    fprs = []
    accuracy = []
    thd = []

    acc_best = 0
    folds = KFold(n=len(loader), n_folds=5, shuffle=False)
    thresh = np.arange(-1.0, 1.0, 0.005)
    for idx, (train, test) in enumerate(folds):
        # find threshold
        best_thresh = find_best_threshold(thresh, dists[train])
        tpr, fpr, acc = eval_acc(best_thresh, dists[test])
        tprs += [tpr]
        fprs += [fpr]
        accuracy += [acc]
        thd.append(best_thresh)

        acc_best = acc if acc > acc_best else acc_best
        print('TPR={:.4f} FPR={:.4f} ACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(tpr),
                                                                              np.mean(fprs),
                                                                              np.mean(accuracy),
                                                                              np.std(accuracy),
                                                                              np.mean(thd)))
        # acc_best = acc is acc > acc_best el
        # Compute ROC curve and ROC area for each class
        # fpr = dict()
        # tpr = dict()
        # roc_auc = dict()
        # for i in range(n_classes):
        #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        #     roc_auc[i] = auc(fpr[i], tpr[i])

    return acc_best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FIW Sphereface Baseline')
    parser.add_argument('--type', '-t', default='bb', type=str, help='relationship type (None processes entire directory)')
    parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
    parser.add_argument('--modelpath', default='finetuned/checkpoint.pth.tar', type=str,
                        help='the pretrained model to point to')
    parser.add_argument('--label_dir', '-l', type=str, default='datasets/FIW/RFIW/val/pairs/',
                            help='Root directory of data (assumed to containing pairs list labels)')
    parser.add_argument('--data_dir', '-d', type=str, default='datasets/FIW/RFIW/val/',
                        help='Root directory of data (assumed to contain valdata)')

    args = parser.parse_args()

    net = net_sphere.sphere20a(classnum=300)

    if cuda:
        net.cuda()

    epoch, bess_acc = TorchTools.load_checkpoint(net, f_weights=args.modelpath)
    csv_file = os.path.join(args.label_dir, args.type + '_val.csv')
    loader = get_val_loader(args.data_dir, csv_file)
    accuracy = validate(net, loader)

    print('Obtained {} on validation set'.format(accuracy))