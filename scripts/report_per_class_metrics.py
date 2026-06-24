"""Load a saved test confusion matrix and print per-class precision, recall, IoU.

Usage:
    python scripts/report_per_class_metrics.py <path_to_test_confmat.pt>
"""

import sys
from pathlib import Path

import torch


def main(cm_path: str) -> None:
    blob = torch.load(cm_path, map_location="cpu")
    cm = blob["confmat"].double()
    names = list(blob["class_names"])[: cm.shape[0]]

    # Convention from src/metrics/semantic.py: confmat[gt, pred]
    # (rows = ground truth, cols = predicted).
    tp = cm.diag()
    gt = cm.sum(dim=1)
    pred = cm.sum(dim=0)

    recall = tp / gt.clamp(min=1)
    precision = tp / pred.clamp(min=1)
    iou = tp / (gt + pred - tp).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)

    print(f"Confusion matrix from: {cm_path}")
    print(f"Total points: {int(cm.sum().item()):,}")
    print()
    print(f"{'class':<16}{'precision':>12}{'recall':>10}{'IoU':>10}{'F1':>10}{'support':>14}")
    print("-" * 72)
    for i, n in enumerate(names):
        print(
            f"{n:<16}{precision[i].item()*100:>11.2f}%"
            f"{recall[i].item()*100:>9.2f}%"
            f"{iou[i].item()*100:>9.2f}%"
            f"{f1[i].item()*100:>9.2f}%"
            f"{int(gt[i].item()):>14,}"
        )
    print()
    print(f"macc (mean recall):    {recall.mean().item()*100:.2f}%")
    print(f"mPrec:                 {precision.mean().item()*100:.2f}%")
    print(f"mIoU:                  {iou.mean().item()*100:.2f}%")
    print(f"OA:                    {(tp.sum()/cm.sum()).item()*100:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
