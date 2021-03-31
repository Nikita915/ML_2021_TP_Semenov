import numpy as np

def confusion_stats(y_true, y_predict):
    FP = y_predict > y_true
    FN = y_predict < y_true
    TP = y_predict.astype(bool) & y_true.astype(bool)
    TN = np.logical_not(y_predict.astype(bool)) & np.logical_not(y_true.astype(bool))
    return {'TP': np.sum(TP), 'TN': np.sum(TN), 'FP': np.sum(FP), 'FN': np.sum(FN)}

def precision(y_true, y_predict):
    cs = confusion_stats(y_true, y_predict)
    summ = cs['TP'] + cs['FP']
    if summ == 0:
        return 1
    return cs['TP'] / summ

def recall(y_true, y_predict):
    cs = confusion_stats(y_true, y_predict)
    return cs['TP'] / (cs['TP'] + cs['FN'])    

def accuracy(y_true, y_predict):
    cs = confusion_stats(y_true, y_predict)
    return (cs['TP'] + cs['TN']) / (cs['TP'] + cs['TN'] + cs['FN'] + cs['FP'])

def f1(y_true, y_predict):
    rec = recall(y_true, y_predict)
    prec = precision(y_true, y_predict)
    return 2 * (rec * prec) / ([prec + rec])

def lift(y_true, y_predict):
    num = precision(y_true, y_predict)
    cs = confusion_stats(y_true, y_predict)
    denum = (cs['TP'] + cs['FN']) / (cs['TP'] + cs['TN'] + cs['FN'] + cs['FP'])
    return num/denum
    
def score(y_true, y_predict, percent=None, function=accuracy):
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    if not percent:
        y_predict = (y_predict[:, 1] >= 0.5).astype(float)
        return function(y_true, y_predict)
    percent = abs(percent)
    assert percent <= 100, 'percent > 100'
    if percent <= 1:
        y_predict = (y_predict[:, 1] >= percent).astype(float)
        return function(y_true, y_predict)
    if percent > 1:
        ind = np.argsort(y_predict[:, 1])[::-1]
        y_predict = np.take_along_axis(y_predict[:, 1], ind, axis=0)
        y_predict = (y_predict >= percent / 100).astype(float)
        y_true = np.take_along_axis(y_true, ind, axis=0)
        score = []
        per = int(len(y_true) / 100)
        for n in np.arange(per, 100 * per, per):
            score.append(function(y_true[:n], y_predict[:n]))
        return np.array(score)

def precision_score(y_true, y_predict, percent=None):
    return score(y_true, y_predict, percent=None, function=precision)

def recall_score(y_true, y_predict, percent=None):
    return score(y_true, y_predict, percent=None, function=recall)

def accuracy_score(y_true, y_predict, percent=None):
    return score(y_true, y_predict, percent=None, function=accuracy)

def lift_score(y_true, y_predict, percent=None):
    return score(y_true, y_predict, percent=None, function=lift)

def f1_score(y_true, y_predict, percent=None):
    return score(y_true, y_predict, percent=None, function=f1)






