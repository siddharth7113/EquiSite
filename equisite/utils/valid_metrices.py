"""Validation metrics used for binary binding-site predictions."""

import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_recall_curve


def CFM_eval_metrics(CFM: np.ndarray) -> tuple[float, float, float, float, float]:
    """
    Cfm eval metrics.

    Parameters
    ----------
    CFM : Any
        Input argument.

    Returns
    -------
    Any
        Function output.
    """
    CFM = CFM.astype(float)
    tn = CFM[0, 0]
    fp = CFM[0, 1]
    fn = CFM[1, 0]
    tp = CFM[1, 1]
    if tp > 0:
        rec = tp / (tp + fn)
    else:
        rec = 0
    if tp > 0:
        pre = tp / (tp + fp)
    else:
        pre = 0
    if tn > 0:
        spe = tn / (tn + fp)
    else:
        spe = 0
    if rec + pre > 0:
        F1 = 2 * rec * pre / (rec + pre)
    else:
        F1 = 0
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0:
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    else:
        mcc = -1
    return rec, pre, F1, spe, mcc


def best_threshold_by_mcc(labels: np.ndarray, preds: np.ndarray) -> tuple[float, float]:
    # Compute precision, recall, and thresholds.
    """
    Best threshold by mcc.

    Parameters
    ----------
    labels : Any
        Input argument.
    preds : Any
        Input argument.

    Returns
    -------
    Any
        Function output.
    """
    precision, recall, thresholds = precision_recall_curve(labels, preds)

    # Initialize the best MCC score and threshold.
    best_mcc = -1
    best_threshold = 0

    # Include all operating points by prepending 0 to the thresholds.
    thresholds = np.insert(thresholds, 0, 0)

    # Iterate over thresholds and compute the corresponding MCC.
    for threshold in thresholds:
        # Convert probabilities to binary predictions at the current threshold.
        binary_preds = (preds >= threshold).astype(int)

        # Compute the confusion matrix.
        tn, fp, fn, tp = confusion_matrix(labels, binary_preds).ravel()

        # Compute MCC for the current threshold.
        mcc = matthews_corrcoef(labels, binary_preds)

        # Update the best MCC and threshold when improved.
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold

    return best_threshold, best_mcc
