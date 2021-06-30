#!/usr/bin/env python3
"""
evalulation
"""
import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics import f1_score
SUBSET2IDS = {
    'easy': [0, 7, 19],
    'medium': [1, 3, 4, 6, 9, 10, 11, 12, 13, 14, 15, 16, 18, 22, 26],
    'hard': [2, 5, 8, 17, 20, 21, 23, 24, 25, 27],
    'object': [0, 3, 10, 11, 12, 16, 23],
    'context': [7, 8],
    "other": [1, 2, 4, 5, 6, 9, 13, 17, 18, 19, 20, 22, 25, 27, 14, 15, 21,  24, 26],
}


def eval_all_metrics(
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    val_targets: List[List[int]],
    test_targets: List[List[int]]
) -> dict:
    """
    compute validation and test results
    args:
        val_scores: np.ndarray of shape (val_num, num_classes),
        test_scores: np.ndarray of shape (test_num, num_classes),
        val_targets: List[List[int]],
        test_targets: List[List[int]]
    """
    # get optimal threshold using val set
    multihot_targets = multihot(val_targets, 28)
    f1_dict = get_best_f1_scores(multihot_targets, val_scores)

    # get results using the threshold found
    multihot_targets = multihot(test_targets, 28)
    test_micro, test_samples, test_macro, test_none = compute_f1(multihot_targets, test_scores, f1_dict["threshold"])
    return {
        "val_micro": f1_dict["micro"], "val_samples": f1_dict["samples"],
        "val_macro": f1_dict["macro"], "val_none": f1_dict["none"],
        "test_micro": test_micro, "test_samples": test_samples,
        "test_macro": test_macro, "test_none": test_none,
    }


def get_best_f1_scores(
    multihot_targets: np.ndarray,
    scores: np.ndarray,
    threshold_end: float = 0.05
) -> Dict[str, float]:
    """
    get the optimal macro f1 score by tuning threshold
    """
    thrs = np.linspace(
        threshold_end, 0.95, int(np.round((0.95 - threshold_end) / 0.05)) + 1,
        endpoint=True
    )
    f1_micros = []
    f1_macros = []
    f1_samples = []
    f1_none = []
    for thr in thrs:
        _micros, _samples, _macros, _none = compute_f1(multihot_targets, scores, thr)
        f1_micros.append(_micros)
        f1_samples.append(_samples)
        f1_macros.append(_macros)
        f1_none.append(_none)

    f1_macros_m = max(f1_macros)
    b_thr = np.argmax(f1_macros)

    f1_micros_m = f1_micros[b_thr]
    f1_samples_m = f1_samples[b_thr]
    f1_none_m = f1_none[b_thr]
    f1 = {}
    f1["micro"] = f1_micros_m
    f1["macro"] = f1_macros_m
    f1["samples"] = f1_samples_m
    f1["threshold"] = thrs[b_thr]
    f1["none"] = f1_none_m
    return f1


def compute_f1(
        multihot_targets: np.ndarray, scores: np.ndarray, threshold: float = 0.5
) -> Tuple[float, float, float]:
    # change scores to predict_labels
    predict_labels = scores > threshold
    predict_labels = predict_labels.astype(np.int)

    # get f1 scores
    f1 = {}
    f1["micro"] = f1_score(
        y_true=multihot_targets,
        y_pred=predict_labels,
        average="micro"
    )
    f1["samples"] = f1_score(
        y_true=multihot_targets,
        y_pred=predict_labels,
        average="samples"
    )
    f1["macro"] = f1_score(
        y_true=multihot_targets,
        y_pred=predict_labels,
        average="macro"
    )
    f1["none"] = f1_score(
        y_true=multihot_targets,
        y_pred=predict_labels,
        average=None
    )
    return f1["micro"], f1["samples"], f1["macro"], f1["none"]


def multihot(x: List[List[int]], nb_classes: int) -> np.ndarray:
    """transform to multihot encoding

    Arguments:
        x: list of multi-class integer labels, in the range
            [0, nb_classes-1]
        nb_classes: number of classes for the multi-hot vector

    Returns:
        multihot: multihot vector of type int, (num_samples, nb_classes)
    """
    num_samples = len(x)

    multihot = np.zeros((num_samples, nb_classes), dtype=np.int32)
    for idx, labs in enumerate(x):
        for lab in labs:
            multihot[idx, lab] = 1

    return multihot.astype(np.int)
