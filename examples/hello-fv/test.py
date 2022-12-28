from __future__ import print_function

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def main():

    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )

    args, unparsed = parser.parse_known_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    test_kwargs = {"batch_size": args.test_batch_size}

    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST("/data", train=False, download=False, transform=transform)

    test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)
    model = Net().to(device)

    model.load_state_dict(torch.load("/weight/weight.ckpt")["state_dict"])

    model.eval()

    y_true, y_pred = [], []
    y_probobility = []

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            pred_list = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(pred_list)

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)

            # Probobilities for roc curve
            for i in range(len(pred_list)):
                pred = pred_list[i]
                y_probobility.append(output[i][pred].cpu().numpy())

    # One-vs-rest precision, recall, f1 score
    precision, recall, f1_score = [], [], []
    multilabel_cf_matrix = metrics.multilabel_confusion_matrix(y_true, y_pred)
    for idx, matrix in enumerate(multilabel_cf_matrix):
        precision.append(np.nan_to_num(matrix[1][1] / (matrix[0][1] + matrix[1][1])))
        recall.append(np.nan_to_num(matrix[1][1] / (matrix[1][1] + matrix[1][0])))
        f1_score.append(
            np.nan_to_num(2 * matrix[1][1] / (2 * matrix[1][1] + matrix[0][1] + matrix[1][0]))
        )

    # Confusion Matrix
    cf_matrix = np.nan_to_num(metrics.confusion_matrix(y_true, y_pred))

    # One-vs-rest ROC
    fpr, tpr = [], []
    for i in range(10):
        fpr_, tpr_, _ = metrics.roc_curve(y_pred, y_probobility, pos_label=i)
        fpr.append(np.nan_to_num(fpr_))
        tpr.append(np.nan_to_num(tpr_))

    # Export json
    result = {
        "metadata": {
            "datasetSize": len(test_loader.dataset),
        },
        "results": {
            "tables": [
                {
                    "title": "Class 1 eval metrics",
                    "labels": ["f1", "precision", "recall"],
                    "values": [f1_score[1], precision[1], recall[1]],
                },
                {
                    "title": "Class 9 eval metrics",
                    "labels": ["f1", "precision", "recall"],
                    "values": [f1_score[9], precision[9], recall[9]],
                },
            ],
            "heatmaps": [
                {
                    "title": "Confusion Matrix",
                    "x-labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                    "y-labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                    "x-axis": "numbers",
                    "y-axis": "numbers",
                    "values": cf_matrix.tolist(),
                },
            ],
            "plots": [
                {
                    "title": "Class 1 roc curve",
                    "labels": ["Class 1"],
                    "x-axis": "fpr",
                    "y-axis": "tpr",
                    "x-values": [[fpr[1].tolist()]],
                    "y-values": [[tpr[1].tolist()]],
                },
                {
                    "title": "Class 9 roc curve",
                    "labels": ["Class 9"],
                    "x-axis": "fpr",
                    "y-axis": "tpr",
                    "x-values": [[fpr[9].tolist()]],
                    "y-values": [[tpr[9].tolist()]],
                },
            ],
        },
    }

    with open("/output/result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    main()
