import numpy as np
import argparse
import net_sphere
import os
from torchtools import cuda, TorchTools, Tensor
from data_loader import get_val_loader
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from torch.autograd import Variable
from src.common.io import sys_home
import math
from scipy import interp

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
import torch
def validate(net, data_loader):
    print('Begin validation')
    net.eval()
    dists = []
    y_labels = []
    distances = []
    for pairs, labels in iter(data_loader):

        # using nets to extract features from pairs and comparing (scoring using cosine distance)
        img_a = Variable(pairs[0]).type(Tensor)
        img_b = Variable(pairs[1]).type(Tensor)

        _, embs_a = net(img_a)
        _, embs_b = net(img_b)

        embs_a = embs_a.data
        embs_b = embs_b.data
        cos_dis = torch.nn.functional.cosine_similarity(embs_a, embs_b)
        distances += list(cos_dis.data.cpu().numpy())

        y_labels += list(labels.numpy())

    dist_array = np.array(distances)
    y_array = np.array(y_labels)

    tprs = []
    fprs = []
    accuracy = []
    thd = []
    acc_best = 0
    folds = KFold(n=len(loader), n_folds=5, shuffle=False)
    aucs = []

    # thresh = np.arange(-1.0, 1.0, 0.005)
    mean_fpr = np.linspace(0, 1, 100)
    i=0
    for idx, (train, test) in enumerate(folds):
        # find threshold
        fpr, tpr, thresholds = roc_curve(y_array[test], dist_array[test])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, float(std_auc)),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    fpr, tpr, thresh = roc_curve(y_array, dist_array)
    roc_auc = auc(fpr, tpr)


    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkred',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

    # Compute micro-average ROC curve and ROC area
    fpr2, tpr2, _ = roc_curve(y_array.ravel(), dist_array.ravel())
    roc_auc2 = auc(fpr2, tpr2)

    plt.plot(fpr2, tpr2, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc2)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    # best_thresh = find_best_threshold(thresh, dists[train])
    # tpr, fpr, acc = eval_acc(best_thresh, dists[test])
    #
    # tprs += [tpr]
    # fprs += [fpr]
    # accuracy += [acc]
    # thd.append(best_thresh)
    #
    # acc_best = acc if acc > acc_best else acc_best
    # print('TPR={:.4f} FPR={:.4f} ACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(tpr),
    #                                                                       np.mean(fprs),
    #                                                                       np.mean(accuracy),
    #                                                                       np.std(accuracy),
    #                                                           np.mean(thd)))

    return acc_best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FIW Sphereface Baseline')
    parser.add_argument('--type', '-t', default='bb', type=str, help='relationship type (None processes entire directory)')
    parser.add_argument('--batch_size', default=256, type=int, help='training batch size')
    parser.add_argument('--modelpath', default='finetuned/checkpoint.pth.tar', type=str,
                        help='the pretrained model to point to')
    parser.add_argument('--label_dir', '-l', type=str, default=sys_home() + '/datasets/FIW/RFIW/val/pairs/',
                            help='Root directory of data (assumed to containing pairs list labels)')
    parser.add_argument('--data_dir', '-d', type=str, default=sys_home() + '/datasets/FIW/RFIW/val/',
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