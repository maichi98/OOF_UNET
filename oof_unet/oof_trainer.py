import torch
import torch.optim as optim
import logging
import time
import os
import matplotlib.pyplot as plt
import numpy as np

from oof_unet import constants


class OofTrainer:

    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 fold,
                 num_epochs=100,
                 lr=1e-3,
                 device='cuda'):

        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs

        # Set up the optimizer and loss function.
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.dir_results = constants.DIR_RESULTS / f"fold_{fold}"
        self.dir_results.mkdir(exist_ok=True, parents=True)

        self.path_log = self.dir_results / f"training_log_{time.strftime('%Y%m%d_%H%M%S')}.log"
        self.path_loss_fig = self.dir_results / f"loss_curve_{time.strftime('%Y%m%d_%H%M%S')}.png"

        # Set up the logger.
        logging.basicConfig(filename=self.path_log, level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s')

        logging.info(f"Training fold {fold} for {self.num_epochs} epochs.")

        self.best_val_loss = float('inf')

        # Lists to store losses for plotting.
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, epoch):

        self.model.train()
        running_loss = 0.0

        for batch in self.train_loader:

            in_field = batch["in_field"].to(self.device)
            mask = batch["mask"].to(self.device)
            inputs = torch.cat([in_field, mask], dim=1)

            targets = batch["out_field"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.compute_masked_mse_loss(outputs, targets, mask)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        msg = f"Epoch [{epoch + 1}/{self.num_epochs}] Validation Loss: {avg_loss:.4f}"
        print(msg)
        logging.info(msg)
        return avg_loss

    def compute_masked_mse_loss(self, outputs, targets, mask, epsilon=1e-8):

        error = (outputs - targets) ** 2
        masked_error = error * mask
        loss = masked_error.sum() / (mask.sum() + epsilon)
        return loss

    def plot_losses(self, epoch):
        epochs = np.arange(1, epoch + 2)
        plt.figure()
        plt.plot(epochs, self.train_losses, label="Training Loss")
        plt.plot(epochs, self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.savefig(self.path_loss_fig)
        plt.close()

    def validate_epoch(self, epoch):

        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:

                in_field = batch["in_field"].to(self.device)
                mask = batch["mask"].to(self.device)
                inputs = torch.cat([in_field, mask], dim=1)

                targets = batch["out_field"].to(self.device)

                outputs = self.model(inputs)
                loss = self.compute_masked_mse_loss(outputs, targets, mask)
                running_loss += loss.item()

        avg_loss = running_loss / len(self.val_loader)
        print(f"Epoch [{epoch + 1}/{self.num_epochs}] Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def save_model(self, epoch, tag):

        if tag == 'best':
            filename = "best_model.pth"
        elif tag == 'every50':
            filename = f"model_epoch_{epoch + 1}_checkpoint.pth"
        else:
            filename = "last_model.pth"

        path = self.dir_results / filename
        torch.save(self.model.state_dict(), path)
        logging.info(f"Saved {tag} model at epoch {epoch + 1} to {path}")

    def train(self):

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch)

            # Save the last model every epoch.
            self.save_model(epoch, tag="last")

            # Save the best model if validation loss improves.
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(epoch, tag="best")
                logging.info(f"New best validation loss {val_loss:.4f} at epoch {epoch + 1}")

            # Save a checkpoint every 50 epochs.
            if (epoch + 1) % 50 == 0:
                self.save_model(epoch, tag="every50")

            # Plot and save the training/validation loss curve after each epoch.
            self.plot_losses(epoch)
