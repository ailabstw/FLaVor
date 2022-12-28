from __future__ import print_function

import argparse
import os
from multiprocessing import Process

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from flavors.cook.log_msg import LogLevel, PackLogMsg
from flavors.cook.servicer import EdgeAppServicer, serve


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


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

    return 100.0 * correct / len(test_loader.dataset)


def run(namespace, trainInitDoneEvent, trainStartedEvent, trainFinishedEvent, logQueue):
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        metavar="N",
        help="number of epochs to train (default: 300)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--no-mps", action="store_true", default=False, help="disables macOS GPU training"
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False, help="quickly check a single pass"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model", action="store_true", default=False, help="For Saving the current Model"
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    logQueue.put(PackLogMsg(LogLevel.INFO, "Load dataset ..."))
    dataset1 = datasets.MNIST("/data", train=True, transform=transform)
    dataset2 = datasets.MNIST("/data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    logQueue.put(PackLogMsg(LogLevel.INFO, "Set model and optimizer ..."))
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    trainInitDoneEvent.set()

    logQueue.put(PackLogMsg(LogLevel.INFO, "Start Training ..."))
    for epoch in range(args.epochs):

        # Wait for the server
        trainStartedEvent.wait()
        trainStartedEvent.clear()

        # Load checkpoint sent from the server
        if epoch != 0 or os.path.exists(namespace.globalModelPath):
            model.load_state_dict(torch.load(namespace.globalModelPath)["state_dict"])

        train(args, model, device, train_loader, optimizer, epoch)
        precision = test(model, device, test_loader)
        scheduler.step()

        # Save checkpoint
        torch.save({"state_dict": model.state_dict()}, "mnist_cnn.ckpt")

        # Save information that the server needs to know
        namespace.metadata = {"epoch": str(epoch), "datasetSize": str(len(dataset1))}
        namespace.metrics = {
            "precision": precision,
            "basic/confusion_tp": -1,  # If N/A or you don't want to track, fill in -1.
            "basic/confusion_fp": -1,
            "basic/confusion_fn": -1,
            "basic/confusion_tn": -1,
            "basic/weight": -1,
        }
        namespace.epoch_path = "mnist_cnn.ckpt"

        # Tell the server that this round of training work has ended.
        trainFinishedEvent.set()


def main(namespace, trainInitDoneEvent, trainStartedEvent, trainFinishedEvent, logQueue):
    try:
        run(namespace, trainInitDoneEvent, trainStartedEvent, trainFinishedEvent, logQueue)
    except Exception as err:
        logQueue.put(PackLogMsg(LogLevel.ERROR, str(err)))


if __name__ == "__main__":

    app_service = EdgeAppServicer()
    app_service.dataPreProcess = None
    app_service.trainingProcess = Process(
        target=main,
        kwargs={
            "namespace": app_service.namespace,
            "trainInitDoneEvent": app_service.trainInitDoneEvent,
            "trainStartedEvent": app_service.trainStartedEvent,
            "trainFinishedEvent": app_service.trainFinishedEvent,
            "logQueue": app_service.logQueue,
        },
    )

    serve(app_service)
