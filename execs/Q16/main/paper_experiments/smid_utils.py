import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torchmetrics

sns.set_style("whitegrid")


def accuracy(y_pred, y_gt, cm):
    y_pred_tag = torch.argmax(y_pred, dim=-1)
    for i in range(len(cm)):
        for j in range(len(cm)):
            cm[i][j] += (y_pred_tag[y_gt == i] == j).sum().int().item()

    correct_results_sum = (y_pred_tag == y_gt).sum().float()
    acc = correct_results_sum / y_gt.shape[0]
    acc = acc * 100
    return acc


def precision_recall(pred, target, pos_label=1):
    assert not target.max() > 1
    if len(pred.shape) > 1:
        pred = torch.argmax(pred, dim=-1)
    pr_curve = torchmetrics.PrecisionRecallCurve(pos_label=pos_label)
    average_precision = torchmetrics.AveragePrecision(pos_label=pos_label)
    precision, recall, thresholds = pr_curve(pred, target)
    avg_precision = average_precision(pred, target)
    return precision, recall, thresholds, avg_precision


def f1(pred, target):
    assert not target.max() > 1
    if len(pred.shape) > 1:
        pred = torch.argmax(pred, dim=-1)
    f1 = torchmetrics.F1(num_classes=2)
    f1_score = f1(pred, target)
    return f1_score


def display_plt(save, file_name):
    if save:
        plt.savefig(file_name, dpi=600)
    else:
        plt.show()
    plt.close()

