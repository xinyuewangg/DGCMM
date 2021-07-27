import os

import numpy as np
import xlsxwriter as xlwt

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


def _prf_divide(numerator, denominator):
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1  # avoid infs/nans
    result = numerator / denominator
    return result


class ResultLogger:

    def __init__(self, tag, num_classes, train_class_count, test_class_count, logdir='.', with_auc=True, verbose=False):
        super().__init__()
        self.with_auc = with_auc
        self.tag = tag
        self.num_classes = num_classes
        os.makedirs(logdir, exist_ok=True)
        self.logdir = logdir
        self.verbose = verbose
        self.results = []
        # training metric
        self.training_loss = []
        self.training_accuracies = []
        self.training_time = []
        # test metric
        self.acs_accuracies = []  # Average Class Specific Accuracy
        self.precisions = []
        self.recalls = []
        self.f_macros = []
        self.f_micros = []
        self.g_macros = []
        self.g_micros = []
        self.specificities = []
        self.accuracies_per_class = []
        self.g_per_class = []
        self.reports = []
        self.test_time = []
        self.many_accuracies = []
        self.median_accuracies = []
        self.low_accuracies = []
        self.accuracies = []
        self.auc_scores = []
        self.train_class_count = np.array(train_class_count)
        self.test_class_count = np.array(test_class_count)
        self.remarks = []

    def add_test_metrics(self, y_true, y_pred, y_score=None, time=0., remark=''):
        if self.with_auc and y_score is None:
            raise ValueError('y_score must be not None.')
        self.test_time.append(time)
        y_true, y_pred = y_true.astype(np.int32), y_pred.astype(np.int32)
        report = classification_report(y_true, y_pred, digits=5, output_dict=True)
        self.reports.append(report)

        cnf_matrix = confusion_matrix(y_true, y_pred)
        if self.verbose:
            print(cnf_matrix)

        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)

        many_acc, median_acc, low_acc = self.shot_acc(TP)
        cs_accuracy = _prf_divide(TP, (cnf_matrix.sum(axis=1)))
        precision = _prf_divide(TP, (TP + FP))
        recall = _prf_divide(TP, (TP + FN))
        specificity = _prf_divide(TN, (FP + TN))
        self.accuracies.append(TP.sum() / self.test_class_count.sum())
        self.many_accuracies.append(many_acc)
        self.median_accuracies.append(median_acc)
        self.low_accuracies.append(low_acc)
        self.accuracies_per_class.append(cs_accuracy)
        self.acs_accuracies.append(cs_accuracy.mean())
        self.precisions.append(precision.mean())
        self.recalls.append(recall.mean())
        self.specificities.append(specificity.mean())

        f1_macro = _prf_divide(2 * precision * recall, precision + recall).mean()
        f1_micro = 2 * TP.sum() / (2 * TP.sum() + FP.sum() + FN.sum())
        self.f_macros.append(f1_macro)
        self.f_micros.append(f1_micro)

        g_marco = ((recall * specificity) ** 0.5).mean()
        g_micro = ((TP.sum() / (TP.sum() + FN.sum())) * (TN.sum() / (TN.sum() + FP.sum()))) ** 0.5
        self.g_macros.append(g_marco)
        self.g_micros.append(g_micro)
        self.g_per_class.append((recall * specificity) ** 0.5)
        if self.with_auc:
            auc_score = roc_auc_score(y_true, y_score, multi_class='ovr')
            self.auc_scores.append(auc_score)
        self.remarks.append(remark)
        return g_marco

    def add_training_metrics(self, loss, accuracy, time=0.):
        self.training_loss.append(loss)
        self.training_accuracies.append(accuracy)
        self.training_time.append(time)

    def save_metrics(self):
        # save evaluation results
        filename = self.tag + '_result' + '.xlsx'
        workbook = xlwt.Workbook(self.logdir + os.sep + filename)

        sheet1 = workbook.add_worksheet('evaluation_metrics')
        sheet2 = workbook.add_worksheet('evaluation_metric_per_class')
        sheet3 = workbook.add_worksheet('training_metrics')
        titles1 = ['rec_ma', 'pre_ma', 'spe_ma', 'acc', 'f_ma', 'f_mi', 'g_ma', 'g_mi', 'time',
                   'many', 'median', 'low', 'acsa', 'auc', 'remark']
        for i, title in enumerate(titles1):
            sheet1.write(0, i, title)
        for i in range(len(self.acs_accuracies)):
            row = i + 1
            sheet1.write(row, 0, self.recalls[i])
            sheet1.write(row, 1, self.precisions[i])
            sheet1.write(row, 2, self.specificities[i])
            sheet1.write(row, 3, self.accuracies[i])
            sheet1.write(row, 4, self.f_macros[i])
            sheet1.write(row, 5, self.f_micros[i])
            sheet1.write(row, 6, self.g_macros[i])
            sheet1.write(row, 7, self.g_micros[i])
            sheet1.write(row, 8, self.test_time[i])
            sheet1.write(row, 9, self.many_accuracies[i])
            sheet1.write(row, 10, self.median_accuracies[i])
            sheet1.write(row, 11, self.low_accuracies[i])
            sheet1.write(row, 12, self.acs_accuracies[i])
            if self.with_auc:
                sheet1.write(row, 13, self.auc_scores[i])
            sheet1.write(row, 14, self.remarks[i])

        row = 0
        for i in range(len(self.acs_accuracies)):
            titles2 = ['epoch ' + str(i), 'accuracy', 'precision', 'recall', 'f1-score', 'g-score', 'support']
            for j, title in enumerate(titles2):
                sheet2.write(row, j, title)
            for j in range(self.num_classes):
                row += 1
                sheet2.write(row, 0, 'class ' + str(j))
                if str(j) in self.reports[i] and j < len(self.accuracies_per_class[i]):
                    sheet2.write(row, 1, self.accuracies_per_class[i][j])
                    sheet2.write(row, 2, self.reports[i][str(j)]['precision'])
                    sheet2.write(row, 3, self.reports[i][str(j)]['recall'])
                    sheet2.write(row, 4, self.reports[i][str(j)]['f1-score'])
                    sheet2.write(row, 5, self.g_per_class[i][j])
                    sheet2.write(row, 6, self.reports[i][str(j)]['support'])
            row += 2

        titles3 = ['loss', 'accuracy', 'time']
        for i, title in enumerate(titles3):
            sheet3.write(0, i, title)
        for i in range(len(self.training_loss)):
            row = i + 1
            sheet3.write(row, 0, self.training_loss[i])
            sheet3.write(row, 1, self.training_accuracies[i])
            sheet3.write(row, 2, self.training_time[i])

        workbook.close()

    def shot_acc(self, class_correct, many_shot_thr=100, low_shot_thr=20):
        many_shot = []
        median_shot = []
        low_shot = []
        for i in range(self.num_classes):
            if self.train_class_count[i] > many_shot_thr:
                many_shot.append((class_correct[i] / self.test_class_count[i]))
            elif self.train_class_count[i] < low_shot_thr:
                low_shot.append((class_correct[i] / self.test_class_count[i]))
            else:
                median_shot.append((class_correct[i] / self.test_class_count[i]))

        if len(many_shot) == 0:
            many_shot.append(0)
        if len(median_shot) == 0:
            median_shot.append(0)
        if len(low_shot) == 0:
            low_shot.append(0)

        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)


if __name__ == '__main__':
    n_classes = 100
    y_true = np.arange(0, n_classes).repeat(1000)
    y_true[100:1000] = 3
    print((y_true == 10).sum())
    y_pred = np.arange(0, n_classes).repeat(1000)
    y_score = np.random.dirichlet([2] * n_classes, len(y_pred))
    np.random.shuffle(y_pred[333:])
    rl = ResultLogger('test1', n_classes, [1000] * n_classes, [1000] * n_classes, with_auc=True)
    results = []
    rl.add_test_metrics(y_true, y_pred, y_score)
    rl.save_metrics()
