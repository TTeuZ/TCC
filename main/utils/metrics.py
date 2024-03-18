from sklearn.metrics import roc_curve, auc
import numpy as np

def get_roc_auc(probs, labels):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    roc = (fpr, tpr, thresholds)
    return (roc, roc_auc)


def get_eer(roc):
    fpr, tpr, thresholds = roc

    fnr = 1 - tpr
    eer_list = np.abs(fpr - fnr)
    eer_threshold_index = np.argmin(eer_list)

    threshold = thresholds[eer_threshold_index]
    eer = eer_list[eer_threshold_index]

    return (threshold, eer)