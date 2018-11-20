from __future__ import print_function

import torch
import torch.optim as optim
from torch.autograd import Variable
# torch.backends.cudnn.benchmark = True

import os
import sys
import datetime
import math
import argparse
import numpy as np

import net_sphere
from data_loader import get_train_loader, get_val_loader

cuda = torch.cuda.is_available()
Tensor = torch.FloatTensor if not cuda else torch.cuda.FloatTensor

ptypes = ['bb', 'fd', 'fs', 'md', 'ms', 'sibs', 'ss']  # FIW pair types
csv_base_path = 'pairs/{}_val.csv'


def printoneline(*argv):
    s = ''
    for arg in argv: s += str(arg) + ' '
    s = s[:-1]
    sys.stdout.write('\r' + s)
    sys.stdout.flush()


def save_model(model, filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)


def dt():
    return datetime.datetime.now().strftime('%H:%M:%S')


def KFold(n, n_folds=5, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[int(i * n / n_folds):int(math.ceil((i + 1) * n / n_folds))]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds


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


def validate(base_dir='val'):
    print('Begin validation')

    net.eval()
    for ptype in ptypes:
        csv_path = os.path.join(base_dir, csv_base_path.format(ptype))

        loader = get_val_loader(base_dir, csv_path)

        dists = []
        for pairs, labels in iter(loader):
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

        folds = KFold(n=len(loader), n_folds=5, shuffle=False)
        thresh = np.arange(-1.0, 1.0, 0.005)
        for idx, (train, test) in enumerate(folds):
            best_thresh = find_best_threshold(thresh, dists[train])
            tpr, fpr, acc = eval_acc(best_thresh, dists[test])
            tprs += [tpr]
            fprs += [fpr]
            accuracy += acc
            thd.append(best_thresh)
        # Compute ROC curve and ROC area for each class
        # fpr = dict()
        # tpr = dict()
        # roc_auc = dict()
        # for i in range(n_classes):
        #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        #     roc_auc[i] = auc(fpr[i], tpr[i])

        print('PTYPE={} TPR={:.4f} FPR={:.4f} ACC={:.4f} std={:.4f} thd={:.4f}'.format(ptype,
                                                                                       np.mean(tprs),
                                                                                       np.mean(fprs),
                                                                                       np.mean(accuracy),
                                                                                       np.std(accuracy),
                                                                                       np.mean(thd)))


def train(epoch, loader, args):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    for inputs, targets in iter(loader):

        if cuda: inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs, _ = net(inputs)
        loss = criterion(outputs, targets)
        lossd = loss.data[0]
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        outputs = outputs[0]  # 0=cos_theta 1=phi_theta
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        printoneline(dt(), 'Te=%d Loss=%.4f | AccT=%.4f%% (%d/%d) %.4f %.2f %d'
                     % (epoch, train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total,
                        lossd, criterion.lamb, criterion.it))
        batch_idx += 1

    print('')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FIW Sphereface Baseline')
    parser.add_argument('--net', '-n', default='sphere20a', type=str)
    parser.add_argument('--lr', default=0.01, type=float, help='inital learning rate')
    parser.add_argument('--n_epochs', default=3, type=int, help='number of training epochs')
    parser.add_argument('--batch_size', default=16, type=int, help='training batch size')
    parser.add_argument('--no_train', action='store_true', help='set to not train')
    parser.add_argument('--finetune', action='store_true', help='set to fine-tune the pretrained model')
    parser.add_argument('--pretrained', default='model/sphere20a_20171020.7z', type=str,
                        help='the pretrained model to point to')
    parser.add_argument('--data_dir', '-d', type=str, default='/Users/josephrobinson/datasets/FIW/RFIW/',
                        help='Root directory of data (assumed to contain traindata and valdata)')


    args = parser.parse_args()
    net = getattr(net_sphere, args.net)(classnum=300)
    model_state = net.state_dict()

    if args.finetune:
        print('Fine-tuning pretrained model at {}'.format(args.pretrained))

        for name, param in net.named_parameters():
            if (name[:4] == 'conv' or name[:4] == 'relu') and name[4] is not '4':
                param.requires_grad = False

        pretrained_state = torch.load(args.pretrained)
        pretrained_state = {k: v for k, v in pretrained_state.items() if
                            k in model_state and v.size() == model_state[k].size()}
        model_state.update(pretrained_state)
        net.load_state_dict(model_state)

    if cuda:
        net.cuda()

    criterion = net_sphere.AngleLoss()
    # train_dir = '/Users/josephrobinson/Downloads/'
    train_dir = args.data_dir + '/train/'
    val_dir = args.data_dir + 'val/'
    # 'train'
    train_loader = get_train_loader(train_dir, batch_size=args.batch_size)

    print('start: time={}'.format(dt()))
    if not args.no_train:
        print('Begin train')
        for epoch in range(args.n_epochs):
            if epoch in [0]:
                if epoch != 0: args.lr *= 0.1  # hardcoded for now (n_epochs = 3)
                params = [x for x in net.parameters() if x.requires_grad]
                optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
            train(epoch, train_loader, args)
            save_model(net, '{}_{}.pth'.format(args.net, epoch))

    validate(val_dir)

    print('finish: time={}\n'.format(dt()))
