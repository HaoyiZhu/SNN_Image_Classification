"""
Model and data related util functions.

Author: Haoyi Zhu
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from IPython.display import HTML

import torch
import torch.nn as nn
import torchvision
import tonic
import tonic.transforms as tonic_transforms
import snntorch as snn

from tonic import CachedDataset
from snntorch.spikevision import spikedata
from snntorch import surrogate
from snntorch import spikeplot as splt
from snntorch import utils as U

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_dataloader_static(
    batch_size: int = 128,
    path: str = "./data/mnist",
    subset: int = 1,
    num_workers: int = 0,
):
    """
    Build dataloaders of MNIST dataset.

    Modified from https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html

    Parameters
    ----------
    batch_size: int
        Batch size.
    path: str
        Path to the data folder.
    subset: int
        Reduce factor of dataset. In each iteration only 1 / subset data.

    Returns
    -------
    Train and test dataloader of MNIST dataset.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
        ]
    )

    mnist_train = datasets.MNIST(path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(path, train=False, download=True, transform=transform)

    # reduce datasets by $subset x to speed up training
    U.data_subset(mnist_train, subset)

    train_loader = DataLoader(
        mnist_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        mnist_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader


def build_dataloader_spike(
    batch_size: int = 128,
    path: str = "./data/N-MNIST",
    num_workers: int = 0,
):
    """
    Build cached dataloaders of NMNIST dataset via tonic.

    Modified from https://snntorch.readthedocs.io/en/stable/tutorials/tutorial_7.html

    Parameters
    ----------
    batch_size: int
        Batch size.
    path: str
        Path to the datset folder.

    Returns
    -------
    Cached train and test dataloader of NMNIST dataset.
    """
    sensor_size = tonic.datasets.NMNIST.sensor_size

    # Denoise removes isolated, one-off events
    # time_window
    frame_transform = tonic_transforms.Compose(
        [
            tonic_transforms.Denoise(filter_time=10000),
            tonic_transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
        ]
    )

    trainset = tonic.datasets.NMNIST(
        save_to=path, transform=frame_transform, train=True
    )
    testset = tonic.datasets.NMNIST(
        save_to=path, transform=frame_transform, train=False
    )

    transform = tonic_transforms.Compose(
        [torch.from_numpy, transforms.RandomRotation([-10, 10])]
    )

    cached_trainset = CachedDataset(
        trainset, transform=transform, cache_path="./cache/nmnist/train"
    )
    cached_train_dataloader = DataLoader(
        cached_trainset,
        batch_size=batch_size,
        collate_fn=tonic.collation.PadTensors(),
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )

    # no augmentations for the testset
    cached_testset = CachedDataset(testset, cache_path="./cache/nmnist/test")
    cached_test_dataloader = DataLoader(
        cached_testset,
        batch_size=batch_size,
        collate_fn=tonic.collation.PadTensors(),
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return cached_train_dataloader, cached_test_dataloader


def build_model(
    device: torch.device,
    slope: int = 25,
    beta: float = 0.5,
    spike: bool = False,
):
    """
    Build a CSNN with architecture of 12C5-MP2-64C5-MP2-1024FC10 using snnTorch and PyTorch.
    - 12C5 is a 5 × 5 convolutional kernel with 12 filters
    - MP2 is a 2 × 2 max-pooling function
    - 1024FC10 is a fully-connected layer that maps 1,024 neurons to 10 outputs

    Ref:
        [1] https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html
        [2] https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html

    Parameters
    ----------
    device: torch.device
        CUDA or CPU device.
    slope: int
        Neuron and simulation parameters.
    beta: float
        Neuron and simulation parameters.
    spike: bool
        Whether the input is spike data or static data.

    Returns
    -------
    CSNN model.
    """
    spike_grad = surrogate.fast_sigmoid(slope=slope)

    if spike:
        m = nn.Sequential(
            nn.Conv2d(2, 12, 5),
            nn.MaxPool2d(2),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(12, 64, 5),
            nn.MaxPool2d(2),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 10),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
        ).to(device)
    else:
        m = nn.Sequential(
            nn.Conv2d(1, 12, 5),
            nn.MaxPool2d(2),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(12, 64, 5),
            nn.MaxPool2d(2),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 10),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
        ).to(device)

    return m


def forward_pass(
    m: Callable,
    data: torch.Tensor,
    num_steps: int = 50,
    spike: bool = False,
):
    """
    Forward pass function of CSNN.

    Ref:
        [1] https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html
        [2] https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html

    Parameters
    ----------
    m: Callable
        CSNN model.
    data: torch.Tensor
        Input data.
    num_steps: int
        Number of steps.
    spike: bool
        Whether the input is spike data or static data.

    Returns
    -------
    Outputs of model forwarding.
    """

    mem_rec = []
    spk_rec = []
    U.reset(m)  # resets hidden states for all LIF neurons in net

    if spike:
        for step in range(data.size(0)):
            spk_out, mem_out = m(data[step])
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)
    else:
        for step in range(num_steps):
            spk_out, mem_out = m(data)
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

    return torch.stack(spk_rec), torch.stack(mem_rec)


def plot_data(train_loss_hist, test_acc_hist, save_path):
    """
    Plot training loss and testing accuracy curves
    """
    fig = plt.figure(facecolor="w")
    plt.plot(train_loss_hist)
    plt.title("Train Set Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(f"{save_path}/train_loss.png")

    fig = plt.figure(facecolor="w")
    plt.plot(test_acc_hist)
    plt.title("Test Set Accuracy")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.savefig(f"{save_path}/test_acc.png")


def spike_counter(m, train_dataloader, cfg, device, num=5, spike=True):
    """
    Plot some spike counter examples.
    """
    m.eval()

    data, targets = next(iter(train_dataloader))
    data = data.to(device)
    targets = targets.to(device)
    spk_rec, mem_rec = forward_pass(m, data, cfg.train.num_steps, spike)

    for idx in range(num):
        fig, ax = plt.subplots(facecolor="w", figsize=(12, 7))
        labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        print(f"The target label is: {targets[idx]}")

        #  Plot spike count histogram
        anim = splt.spike_count(
            spk_rec[:, idx].detach().cpu(),
            fig,
            ax,
            labels=labels,
            animate=True,
            interpolate=4,
        )

        HTML(anim.to_html5_video())
        anim.save(f"./exp/{cfg.exp_id}/spike_bar_{idx}_target{targets[idx]}.gif")
