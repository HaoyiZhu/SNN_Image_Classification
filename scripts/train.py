"""
Training script for SNN on NMNIST.

Author: Haoyi Zhu

Ref:
    [1] https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html
"""

import hydra

from tqdm import tqdm

import torch
from snntorch import functional as SF
from snntorch import backprop

from snn_nmnist.utils import *
from snn_nmnist.logger import *


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def batch_accuracy(test_loader, m, num_steps):
    with torch.no_grad():
        total = 0
        acc = 0
        m.eval()

        test_loader = iter(test_loader)
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)

            spk_rec, _ = forward_pass(m, data, num_steps)

            acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)

    return acc / total


@hydra.main(config_path="../configs", config_name="default")
def main(cfg):
    logger = init_logger(cfg)
    logger.info("******************************")
    logger.info(cfg)
    logger.info("******************************")

    train_dataloader, test_dataloader = build_dataloader(
        batch_size=cfg.train.batch_size, path=cfg.data.root, subset=cfg.data.subset, num_workers=cfg.data.num_workers
    )

    m = build_model(device=device, slope=cfg.model.slope, beta=cfg.model.beta)

    optimizer = torch.optim.Adam(m.parameters(), lr=cfg.train.lr, betas=(0.9, 0.999))
    criterion = SF.ce_rate_loss()

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.train.lr_step, gamma=cfg.train.lr_factor
    )

    train_loss_hist = []
    test_acc_hist = []

    for epoch in range(cfg.train.num_epochs):
        current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
        logger.info(
            f"############# Starting Epoch {epoch} | LR: {current_lr} #############"
        )

        train_loss = backprop.BPTT(
            m,
            train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            num_steps=cfg.train.num_steps,
            time_var=False,
            device=device,
        )
        logger.info(f"Epoch {epoch} | Train loss: {train_loss.item()}")
        train_loss_hist.append(train_loss.item())

        lr_scheduler.step()

        test_acc = batch_accuracy(test_dataloader, m, cfg.train.num_steps)
        test_acc_hist.append(test_acc)
        logger.info(f"Epoch {epoch} | Test acc: {test_acc}")

        torch.save(m.state_dict(), f"./exp/{cfg.exp_id}/model_{epoch}.pth")

    plot_data(train_loss_hist, test_acc_hist, f"./exp/{cfg.exp_id}")

    spike_counter(m, train_dataloader, cfg, device, 5)


if __name__ == "__main__":
    main()
