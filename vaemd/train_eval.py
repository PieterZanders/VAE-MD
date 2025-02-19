import torch
import numpy as np
import os
import time
from torch.optim.lr_scheduler import StepLR



def train_model(model,
                train_dataloader,
                valid_dataloader,
                optimizer,
                loss_function,
                hparams,
                save_folder,
                device):
    losses = {
        "train": {"total": [], "recon": [], "kl": []},
        "valid": {"total": [], "recon": [], "kl": []}
    }

    best_valid_loss = float('inf')

    scheduler = StepLR(optimizer,
                       step_size=hparams["step_size"],
                       gamma=hparams["gamma"])

    print("Start Training...")

    start_time = time.time()

    for epoch in range(1, hparams["epochs"]+1):


        train_loss = train_step(model,
                                train_dataloader,
                                optimizer,
                                loss_function,
                                hparams["beta"],
                                device)

        valid_loss = valid_step(model,
                                valid_dataloader,
                                loss_function,
                                hparams["beta"],
                                device)

        for key in losses["train"]:
            losses["train"][key].append(train_loss[key].item())
            losses["valid"][key].append(valid_loss[key].item())

        scheduler.step()
        epoch_duration = time.time() - start_time

        if epoch % 10 == 0 or epoch == 1:

            print(
                f"Epoch {epoch}/{hparams['epochs']} | "
                f"Train Loss: {train_loss['total']:.4f} , "
                f"{hparams['loss_function']['reconstruction']}: {train_loss['recon']:.4f} , "
                f"kl: {train_loss['kl']:.4f} | "
                f"Valid Loss: {valid_loss['total']:.4f} , "
                f"{hparams['loss_function']['reconstruction']}: {valid_loss['recon']:.4f} , "
                f"kl: {valid_loss['kl']:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.5f} | "
                f"Duration: {epoch_duration:.2f}s"
            )

            start_time = time.time()

        if valid_loss["total"] < best_valid_loss:
            best_valid_loss = valid_loss["total"]
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_folder, 'best_model.pth'))

    print(f"Model from Epoch {best_epoch+1} saved as best_model.pth")

    return losses

def train_step(model, train_dataloader, optimizer, loss_function, beta, device):
    model.train()
    losses = {"total": [], "recon": [], "kl": []}

    for data in train_dataloader:

        data = data.to(device)

        output, _, mu, log_var = model(data)

        recon_loss, kl_divergence = loss_function(output, data, mu, log_var)

        total_loss = recon_loss + beta * kl_divergence

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        losses["total"].append(total_loss.item())
        losses["recon"].append(recon_loss.item())
        losses["kl"].append(kl_divergence.item())

    return {key: torch.tensor(values).mean() for key, values in losses.items()}

def valid_step(model, valid_dataloader, loss_function, beta, device):
    model.eval()
    losses = {"total": [], "recon": [], "kl": []}

    with torch.no_grad():
        for data in valid_dataloader:

            data = data.to(device)

            output, _, mu, log_var = model(data)

            recon_loss, kl_divergence = loss_function(output, data, mu, log_var)

            total_loss = recon_loss + beta * kl_divergence

            losses["total"].append(total_loss.item())
            losses["recon"].append(recon_loss.item())
            losses["kl"].append(kl_divergence.item())

    return {key: torch.tensor(values).mean() for key, values in losses.items()}

def evaluate_model(model, dataloader, loss_function, beta, device):

    losses = {"total": [], "recon": [], "kl": []}
    latent_data = {"z": [], "mu": [], "log_var": []}
    x_hat = []

    model.eval()

    with torch.no_grad():
        for data in dataloader:

            data = data.to(device)
            output, latent, means, log_std = model(data)

            recon_loss, kl_divergence = loss_function(output, data, means, log_std)

            total_loss = recon_loss + beta * kl_divergence

            losses["total"].append(total_loss.item())
            losses["recon"].append(recon_loss.item())
            losses["kl"].append(kl_divergence.item())

            latent_data["z"].append(latent.cpu().numpy())
            latent_data["mu"].append(means.cpu().numpy())
            latent_data["log_var"].append(log_std.cpu().numpy())
            x_hat.append(output.cpu().numpy())

    for key in latent_data:
        latent_data[key] = np.concatenate(latent_data[key], axis=0)
        latent_data[key] = np.reshape(latent_data[key], (-1, model.latent_dim))

    x_hat = np.concatenate(x_hat, axis=0)
    x_hat = np.reshape(x_hat, (-1, model.input_dim))

    return {key: torch.tensor(values).mean() for key, values in losses.items()}, latent_data, x_hat
