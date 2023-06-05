from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn,optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from utils import *
import argparse

class NN(nn.Module):
    def __init__(self, num_classes):
        super(NN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (3,3), padding=(1,1), stride=(1,1)),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding=(1,1), stride=(1,1)),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3,3), padding=(1,1), stride=(1,1)),
            nn.BatchNorm2d(num_features = 128),
            nn.ReLU(),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3,3), padding=(1,1), stride=(1,1)),
            nn.BatchNorm2d(num_features = 128),           
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (3,3), padding=(1,1), stride=(1,1)),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding=(1,1), stride=(1,1)),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding=(1,1), stride=(1,1)),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.block4 = nn.Sequential(
            nn.Linear(in_features = 256*4*4, out_features = 256),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = 128),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 10)
        )
        self.flatten = nn.Flatten()
        self.softmax = F.softmax
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        logits = self.block4(x)
        pred = self.softmax(logits, dim=1)

        return logits,pred

def train_loop(model, train_loader, optimizer, loss_fn, train_loss_list, train_acc_list, device):
    size = len(train_loader.dataset)
    train_loss, correct = 0, 0
    model.train()
    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        logits, pred = model.forward(x)
        y_pred = pred.argmax(dim=1)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(x)
        correct += (y_pred == y).type(torch.float).sum().item()
        if i % 100 == 0:
            loss, current = loss.item(), (i+1) * len(x)
            print(f'Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')
    train_loss /= size
    correct /= size
    train_loss_list.append(train_loss)
    train_acc_list.append(correct)

def test_loop(model, test_loader, loss_fn, device, test_loss_list, test_acc_list):
    size = len(test_loader.dataset)
    test_loss, correct = 0,0
    model.eval()

    with torch.no_grad():
        for x,y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, pred = model.forward(x)
            y_pred = pred.argmax(dim=1)

            test_loss += loss_fn(pred,y).item() * len(x)
            correct += (y_pred == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size

    test_loss_list.append(test_loss)
    test_acc_list.append(correct)

    print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')

def parse_args():
    """parse arguments. You can add other arguments if needed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true",
        help="flag to decide whether to train")
    parser.add_argument("--epochs", type=int, default=50,
        help="training epochs")
    parser.add_argument("--seed", type=int, default=3312,
        help="seed of the experiment")
    parser.add_argument("--lr", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--batch_size", type=int, default=64,
        help="the batch size training samples and test samples")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists("./output"):
        os.mkdir("./output")
    if not os.path.exists("./checkpoints"):
        os.mkdir("./checkpoints")
    version = "v1"
    output_root = f"./output/{version}"
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trans = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(32),
        transforms.ToTensor()
    ])

    if args.train or not os.path.exists(f"./checkpoints/{version}.pt"):
        model = NN(num_classes=10)
        model = model.to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        train_dataset = datasets.FashionMNIST(root='./dataset', train=True,
                                            transform=trans, download=True)
        test_dataset = datasets.FashionMNIST(root='./dataset', train=False,
                                            transform=trans, download=True)

        train_loader = DataLoader(dataset=train_dataset,
                                batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset,
                                batch_size=args.batch_size, shuffle=True)

        train_acc_list, test_acc_list = [], []
        train_loss_list, test_loss_list = [], []
        for t in range(args.epochs):
            print(f'Epoch {t+1}\n-------------------------------')
            train_loop(model, train_loader, optimizer, loss_fn, train_loss_list, train_acc_list, device)
            test_loop(model, test_loader, loss_fn, device, test_loss_list, test_acc_list)
        print('Done!')

        if not os.path.exists(output_root):
            os.mkdir(output_root)

        plot_curveII(train_acc_list, test_acc_list, "Train and Test Accuracy",  f"{output_root}/train_test_acc.pdf")
        plot_curveII(train_loss_list, test_loss_list, "Train and Test Loss",  f"{output_root}/train_test_loss.pdf")
    else:
        model = NN(num_classes=10)
        model.load_state_dict(torch.load(f"./checkpoints/{version}.pt"))
        model.eval()

        test_dataset = datasets.FashionMNIST(root='./dataset', train=False,
                                            transform=trans, download=True)
        test_loader_no_shuffle = DataLoader(dataset=test_dataset,
                                batch_size=args.batch_size, shuffle=False)
        conv_features = []
        relu_features = []
        final_features = []
        def conv_hook_forward(module, fea_in, fea_out):
            conv_features.append(nn.Flatten()(fea_out))
            return None
        def relu_hook_forward(module, fea_in, fea_out):
            relu_features.append(nn.Flatten()(fea_out))
            return None

        h_conv = model.block3[6].register_forward_hook(hook=conv_hook_forward)
        h_relu = model.block3[8].register_forward_hook(hook=relu_hook_forward)

        with torch.no_grad():
            for x,y in test_loader_no_shuffle:
                logits, pred = model.forward(x)
                final_features.append(logits)

        conv_features = torch.cat(conv_features, dim = 0).numpy()
        relu_features = torch.cat(relu_features, dim = 0).numpy()
        final_features = torch.cat(final_features, dim = 0).numpy()
        
        pca = PCA()
        tsne = tSNE()

        # Last conv layer
        pca.fit(conv_features)
        pca.visualization(np.array([label for _, label in test_dataset]), savepth=f"{output_root}/conv_PCA.pdf")
        tsne.fit(conv_features, pca.pca_results, speed='fast') # Directly use sklearn.manifold.TSNE, if you want to test my codes, specifying `speed='naive'`
        tsne.visualization(np.array([label for _, label in test_dataset]), savepth=f"{output_root}/conv_tSNE.pdf")
        
        # Last RELU layer
        pca.fit(relu_features)
        pca.visualization(np.array([label for _, label in test_dataset]), savepth=f"{output_root}/relu_PCA.pdf")
        tsne.fit(relu_features, pca.pca_results, speed='fast') # Directly use sklearn.manifold.TSNE, if you want to test my codes, specifying `speed='naive'`
        tsne.visualization(np.array([label for _, label in test_dataset]), savepth=f"{output_root}/relu_tSNE.pdf")

        # Final layer
        pca.fit(final_features)
        pca.visualization(np.array([label for _, label in test_dataset]), savepth=f"{output_root}/final_PCA.pdf")
        tsne.fit(final_features, pca.pca_results, speed='fast') # Directly use sklearn.manifold.TSNE, if you want to test my codes, specifying `speed='naive'`
        tsne.visualization(np.array([label for _, label in test_dataset]), savepth=f"{output_root}/final_tSNE.pdf")

