from __future__ import annotations

import numpy as np


def semantic_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    if target.shape != prediction.shape:
        raise ValueError(
            f"semantic target and prediction shapes must match exactly: "
            f"{target.shape} != {prediction.shape}"
        )
    if target.size == 0:
        raise ValueError("semantic target and prediction must not be empty")
    labels = np.union1d(target, prediction)
    ious, f1s = [], []
    metrics: dict[str, float] = {}
    for label in labels:
        true_positive = np.sum((target == label) & (prediction == label))
        false_positive = np.sum((target != label) & (prediction == label))
        false_negative = np.sum((target == label) & (prediction != label))
        union = true_positive + false_positive + false_negative
        iou = float(true_positive / union) if union else float("nan")
        denom = 2 * true_positive + false_positive + false_negative
        f1 = float(2 * true_positive / denom) if denom else float("nan")
        metrics[f"semantic/iou_class_{int(label)}"] = iou
        ious.append(iou)
        f1s.append(f1)
    metrics["semantic/miou"] = float(np.nanmean(ious))
    metrics["semantic/macro_f1"] = float(np.nanmean(f1s))
    metrics["semantic/accuracy"] = float(np.mean(target == prediction))
    if len(prediction) > 1:
        metrics["semantic/label_flip_rate"] = float(np.mean(prediction[1:] != prediction[:-1]))
    return metrics
