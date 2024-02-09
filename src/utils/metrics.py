import numpy as np

def accuracy(labels, preds):
    return (np.array(labels) == np.array(preds)).sum() / len(labels)

def f1(actual, predicted, label):

    """ A helper function to calculate f1-score for the given `label` """

    # F1 = 2 * (precision * recall) / (precision + recall)
    tp = np.sum((actual==label) & (predicted==label))
    fp = np.sum((actual!=label) & (predicted==label))
    fn = np.sum((predicted!=label) & (actual==label))
    
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

def f1_micro(actual, predicted):
    # `macro` f1- unweighted mean of f1 per label
    p_r_f1 = np.mean([f1(actual, predicted, label) for label in np.unique(actual)], axis=0)
    return p_r_f1[0], p_r_f1[1], p_r_f1[-1]
