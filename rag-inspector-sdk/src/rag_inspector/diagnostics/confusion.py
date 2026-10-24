from __future__ import annotations

from typing import Any, Sequence


def confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    # Determine unique classes deterministically
    present_labels = sorted(set(list(y_true) + list(y_pred)))
    if labels is None:
        final_labels = present_labels
    else:
        # Preserve user-specified ordering, then append any extra observed labels
        final_labels = list(labels)
        for lab in present_labels:
            if lab not in final_labels:
                final_labels.append(lab)

    label_to_idx = {l: i for i, l in enumerate(final_labels)}
    n = len(final_labels)
    matrix = [[0] * n for _ in range(n)]

    for t, p in zip(y_true, y_pred):
        i = label_to_idx.get(t)
        j = label_to_idx.get(p)
        if i is not None and j is not None:
            matrix[i][j] += 1

    return {"labels": final_labels, "matrix": matrix}


def classification_report(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels)
    final_labels = cm["labels"]
    matrix = cm["matrix"]
    n_classes = len(final_labels)

    per_class: dict[str, dict[str, Any]] = {}
    f1_scores = []
    total_samples = len(y_true)

    for i, lab in enumerate(final_labels):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(n_classes)) - tp
        fn = sum(matrix[i][c] for c in range(n_classes)) - tp
        support = sum(matrix[i][c] for c in range(n_classes))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[lab] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        # Include classes with support > 0 or all present classes
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return {
        "labels": final_labels,
        "total_samples": total_samples,
        "per_class": per_class,
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": matrix,
    }
