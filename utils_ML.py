import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import itertools
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from itertools import cycle
from scipy import interp
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(results['mean_test_score'][candidate], results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_confusion(Y_testing, Y_hat, classes, key):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(Y_testing, Y_hat)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['Class {}'.format(k) for k in range(len(classes))], title=key + ' Confusion matrix, without normalization')
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['Class {}'.format(k) for k in range(len(classes))], normalize=True, title=key + ' Confusion matrix, with normalization')
    plt.draw()


def print_and_get_accuracy(Y_testing, Y_hat):
    res = 100 * accuracy_score(Y_testing, Y_hat)
    print('Accuracy : {}'.format('{:.2f}'.format(res)))
    return res


def print_and_get_precision_recall_fscore_support(Y_testing, Y_hat):
    precision, recall, fscore, support = score(Y_testing, Y_hat)
    print('Precision: {}'.format(['{:.2f}'.format(100 * x) for x in precision]))
    print('Recall   : {}'.format(['{:.2f}'.format(100 * x) for x in recall]))
    print('Fscore   : {}'.format(['{:.2f}'.format(100 * x) for x in fscore]))
    print('Support  : {}'.format(['{:.2f}'.format(100 * x) for x in support]))
    return precision, recall, fscore, support


def print_and_get_macro_micro_weighted_fscore(Y_testing, Y_hat):
    macro = f1_score(Y_testing, Y_hat, average='macro')
    micro = f1_score(Y_testing, Y_hat, average='micro')
    weighted = f1_score(Y_testing, Y_hat, average='weighted')

    print('Fscore Macro   : {}'.format('{:.2f}'.format(100 * macro)))
    print('Fscore Micro   : {}'.format('{:.2f}'.format(100 * micro)))
    print('Fscore Weighted: {}'.format('{:.2f}'.format(100 * weighted)))

    return macro, micro, weighted


def print_and_get_classification_report(Y_testing, Y_hat, classes):
    res = classification_report(Y_testing, Y_hat, target_names=['Class {}'.format(k) for k in range(len(classes))], digits=4)
    print(res)
    return res


def plot_roc(Y_testing, Y_hat, classes, key):
    Y_testing_onehot = np.array([[1, 0] if y == 0 else [0, 1] for y in Y_testing.tolist()])
    Y_hat_onehot = np.array([[1, 0] if y == 0 else [0, 1] for y in Y_hat.tolist()])

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(Y_testing_onehot[:, i], Y_hat_onehot[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_testing_onehot.ravel(), Y_hat_onehot.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(classes)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'])
    for i, color in zip(range(len(classes)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(key + ' ROC')
    plt.legend(loc="lower right")
    plt.draw()


def plot_prec_rec_curve(Y_testing, Y_hat, classes, key):
    Y_testing_onehot = np.array([[1, 0] if y == 0 else [0, 1] for y in Y_testing.tolist()])
    Y_hat_onehot = np.array([[1, 0] if y == 0 else [0, 1] for y in Y_hat.tolist()])

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve(Y_testing_onehot[:, i], Y_hat_onehot[:, i])
        average_precision[i] = average_precision_score(Y_testing_onehot[:, i], Y_hat_onehot[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_testing_onehot.ravel(), Y_hat_onehot.ravel())
    average_precision["micro"] = average_precision_score(Y_testing_onehot, Y_hat_onehot, average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))

    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    fig = plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(len(classes)), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))

    #fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25, left=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(key + ' Precision-Recall curve')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

    plt.draw()


def tune(clf, X, Y, param_dist, n_iter_search=3):
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=1, cv=3)
    random_search.fit(X, Y)
    report(random_search.cv_results_)
    return random_search.cv_results_

