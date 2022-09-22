import os

import pandas as pd
import torch

from configs.config import CFG_DICT
from src.loss import get_loss_function

SAVE_PATH = "./logs/{model_type}/{dir_name}".format(
    model_type=CFG_DICT["TRAIN"]["TYPE"],
    dir_name=CFG_DICT["LOG"]["SAVE_DIR_NAME"],
)


def train(
    epoch: int,
    model,
    optimizer: torch.optim,
    DEVICE: str,
    train_loader: torch.utils.data.DataLoader,
):

    loss_function = get_loss_function(CFG_DICT["TRAIN"]["LOSS_FUNC"])
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):

        X, Y = data
        X = X.to(DEVICE, dtype=torch.float)
        Y = Y.to(DEVICE, dtype=torch.float)

        optimizer.zero_grad()
        _, mu, logvar, Y_hat = model(X, Y)
        kld, recon, loss = loss_function(
            X,
            Y_hat,
            mu,
            logvar,
        )

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    sample_av = train_loss / len(train_loader.dataset)
    if epoch % CFG_DICT["TRAIN"]["LOG_INTERVAL"] == 0:
        print("====> Epoch: {} loss: {:.4f}".format(epoch, sample_av))

    return model, sample_av


def test(
    epoch: int, model, DEVICE: str, test_loader: torch.utils.data.DataLoader
):

    loss_function = get_loss_function(CFG_DICT["TRAIN"]["LOSS_FUNC"])
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):

            X, Y = data
            X = X.to(DEVICE, dtype=torch.float)
            Y = Y.to(DEVICE, dtype=torch.float)

            _, mu, logvar, Y_hat = model(X, Y)
            kld, recon, loss = loss_function(
                X,
                Y_hat,
                mu,
                logvar,
            )

            test_loss += loss.item()

    sample_av = test_loss / len(test_loader.dataset)
    if epoch % CFG_DICT["TRAIN"]["LOG_INTERVAL"] == 0:
        print("====> Test set loss: {:.4f}".format(sample_av))

    return sample_av


def fit(model, dataloaders, optimizer: torch.optim, device):

    train_loader, test_loader = dataloaders

    train_losss, val_losss = [], []
    best_loss = 1e9

    for epoch in range(CFG_DICT["TRAIN"]["EPOCHS"] + 1):
        model, train_loss = train(
            epoch, model, optimizer, device, train_loader
        )
        val_loss = test(epoch, model, device, test_loader)
        train_losss.append(train_loss)
        val_losss.append(val_loss)

        if best_loss > val_loss:
            best_loss = val_loss

            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "model": model,
            }

            torch.save(state, os.path.join(SAVE_PATH, "best_state.pkl"))

            df = pd.DataFrame(
                list(zip(train_losss, val_losss)),
                columns=["TrainLoss", "ValLoss"],
            )

            df.to_csv(os.path.join(SAVE_PATH, "loss.csv"))

        if epoch % CFG_DICT["TRAIN"]["LOG_INTERVAL"] == 0:

            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "model": model,
            }

            torch.save(
                state,
                os.path.join(
                    SAVE_PATH,
                    "{epoch}_of_{max_epoch}.pkl".format(
                        epoch=epoch, max_epoch=CFG_DICT["TRAIN"]["EPOCHS"]
                    ),
                ),
            )
