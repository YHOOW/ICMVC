from sklearn.metrics import v_measure_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch
import os
# 去除 DeprecationWarning等警告
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def evaluate(label, pred):
    nmi = v_measure_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return nmi, ari, acc, pur

def inference(loader, model, device, view, data_size):
    model.eval()
    commonH = []
    preds = []
    labels_vector = []

    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            _, commonh, commonq, pred = model.forward_test(xs)

            commonh = commonh.detach()
            commonH.extend(commonh.cpu().detach().numpy())

            pred = pred.detach()
            preds.extend(pred.cpu().detach().numpy())

        labels_vector.extend(y.numpy())
    labels_vector = np.array(labels_vector).reshape(data_size)

    commonH = np.array(commonH)
    preds = np.array(preds)

    return labels_vector, commonH, preds

def valid(args, model, device, dataset, view, data_size, class_num):
    test_loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
        )
    labels_vector, commonH, preds = inference(test_loader, model, device, view, data_size)

    print('Clustering results:')
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    y_pred = kmeans.fit_predict(commonH)

    # Feature
    nmi_f, ari_f, acc_f, pur_f = evaluate(labels_vector, y_pred)
    print('Feature=====ACC = {:.4f} NMI = {:.4f} PUR={:.4f} ARI = {:.4f}\n'.format(acc_f, nmi_f, pur_f, ari_f))

    # Cluster
    nmi_c, ari_c, acc_c, pur_c = evaluate(labels_vector, preds)
    print('Cluster=====ACC = {:.4f} NMI = {:.4f} PUR={:.4f} ARI = {:.4f}\n'.format(acc_c, nmi_c, pur_c, ari_c))

    with open(os.path.join(args.record_path, 'fine_tune_loss_log.txt'), 'a') as file:
        file.write("Feature=====Clustering results: ACC={:.4f} NMI={:.4f} PUR={:.4f} ARI = {:.4f}\n".format(acc_f, nmi_f, pur_f, ari_f))
        file.write("Cluster=====Clustering results: ACC={:.4f} NMI={:.4f} PUR={:.4f} ARI = {:.4f}\n".format(acc_c, nmi_c, pur_c, ari_c))


def effect_valid(model, device, dataset, view, data_size, class_num):
    test_loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
        )
    labels_vector, commonH, preds = inference(test_loader, model, device, view, data_size)
    print('Clustering results:')
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    y_pred = kmeans.fit_predict(commonH)

    # Feature
    nmi_f, ari_f, acc_f, pur_f = evaluate(labels_vector, y_pred)
    # Cluster
    nmi_c, ari_c, acc_c, pur_c = evaluate(labels_vector, preds)

    return nmi_f, ari_f, acc_f, pur_f, nmi_c, ari_c, acc_c, pur_c