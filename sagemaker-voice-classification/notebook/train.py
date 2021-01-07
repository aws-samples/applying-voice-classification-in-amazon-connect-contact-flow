"""

# Reference papers

- [Very Deep Convolutional Neural Networks for Raw Waveforms]
  (https://arxiv.org/abs/1610.00087)

"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from pathlib import Path
from coswara_dataset import CoswareDataset
from coswara_model import NetM3
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, fbeta_score, roc_auc_score
## oversampling to handle imbalance data set
import numpy as np
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)

def print_files_in_path(path):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            files.append(os.path.join(r, file))

    for f in files:
        print(f)


def train(model, epoch, train_loader, device, optimizer, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        ## oversampling
        data_resampled, target_resampled = ros.fit_resample(np.squeeze(data), target)
        data = torch.from_numpy(data_resampled)
        data = data.unsqueeze_(-2)
        target = torch.tensor(target_resampled)
        
        data, target = data.to(device), target.to(device)
        output = model(data)
        output = output.permute(1, 0, 2)[0]  # original output dimensions are batchSizex1x10
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        accuracy = accuracy_score(target_resampled, pred.cpu().numpy().flatten())
        loss = F.nll_loss(output, target)  # the loss functions expects a batchSizex10 input
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {}, Loss: {:.4f}, Accuracy: {:.4f}".format(
                    epoch, loss, accuracy
                )
            )


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    actuals = []
    predictions = []
    prediction_probs = []
    with torch.no_grad():
        for data, target in test_loader:
            ## oversampling
            data_resampled, target_resampled = ros.fit_resample(np.squeeze(data), target)
            data = torch.from_numpy(data_resampled)
            data = data.unsqueeze_(-2)
            target = torch.tensor(target_resampled)

            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output.permute(1, 0, 2)[0]
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            pred_prob = output.cpu().detach().numpy()[:,1] # get the log-probability for the second class that will be used to calculate prediction probability later using numpy exponential function
            actuals.extend(target.cpu().numpy())
            predictions.extend(pred.cpu().numpy().flatten())
            prediction_probs.extend(pred_prob)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(target.cpu().numpy())

    test_loss /= total
    accuracy = accuracy_score(actuals, predictions)
    rocauc = roc_auc_score(actuals, np.exp(prediction_probs))
    precision = precision_score(actuals, predictions, average='weighted')
    recall = recall_score(actuals, predictions, average='weighted')
    f1 = f1_score(actuals, predictions, average='weighted')
    f2 = fbeta_score(actuals, predictions, average='weighted', beta=0.5)
    
    print(
        "Test set: Average loss: {:.4f}, F1: {:.4f}, F2: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, ROCAUC: {:.4f}, Accuracy: {:.4f}, corrected prediction ratio: {}/{}".format(
            test_loss, f1, f2, precision, recall, rocauc, accuracy, correct, total
        )
    )
    print("\n")
    return test_loss, accuracy


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NetM3()
    if torch.cuda.device_count() > 1:
        print("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def save_model(model, model_dir):
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), path)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64, help="train batch size")
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="test batch size",
    )
    parser.add_argument("--epochs", type=int, default=2, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.01, help="Learning rate step gamma")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="Optimizer regularization")
    parser.add_argument("--stepsize", type=int, default=5, help="Step LR size")
    parser.add_argument("--model", type=str, default="m3")
    parser.add_argument("--num-workers", type=int, default=30)
    parser.add_argument("--csv-file", type=str, default="breathing-deep-metadata.csv")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--localpath", type=str, default="data")

    # Container environment
    parser.add_argument("--model-dir", type=str, default=os.getenv("SM_MODEL_DIR", "./"))
    if os.getenv("SM_HOSTS") is not None:
        # parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
        parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
        # parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
        # print_files_in_path(os.environ["SM_CHANNEL_TRAINING"])

    args = parser.parse_args()
    print(args)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # On SageMaker, data is mounted to SM_CHANNEL_TRAINING, update channel to use sample/full dataset
    if os.getenv("SM_HOSTS") is not None:
        print("Running on sagemaker")
        datapath = Path(args.data_dir)
        csv_path = datapath / args.csv_file
        file_path = datapath
    # Local, use smaller data subset for testing first
    else:
        print("Running on local")
        full_filepath = Path(__file__).resolve()
        parent_path = full_filepath.parent.parent
        csv_path = parent_path / args.localpath / "breathing-deep-metadata.csv"
        file_path = parent_path / args.localpath

    print("csv_path", csv_path)
    print("file_path", file_path)
    kwargs = {"num_workers": args.num_workers, "pin_memory": True} if torch.cuda.is_available() else {}
    print(kwargs)

    dataset = CoswareDataset(
        csv_path=csv_path,
        file_path=file_path,
        new_sr=8000,
        audio_len=20,
        sampling_ratio=5,
    )
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"train_size: {train_size}, test_size:{test_size}")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    print("Loading model:", args.model)
    if args.model == "m3":
        model = NetM3()
    else:
        model = NetM3()

    if torch.cuda.device_count() > 1:
        print("There are {} gpus".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    log_interval = args.log_interval

    for epoch in range(1, args.epochs + 1):
        print("Learning rate:", scheduler.get_last_lr()[0])
        train(model, epoch, train_loader, device, optimizer, log_interval)
        loss, accuracy = test(model, test_loader, device)
        scheduler.step()

    save_model(model, args.model_dir)


if __name__ == "__main__":
    main()
