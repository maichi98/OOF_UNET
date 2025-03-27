import torch
import torch.optim as optim
import logging
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter


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
        self.scaler = GradScaler()

        self.dir_results = constants.DIR_RESULTS / f"fold_{fold}"
        self.dir_results.mkdir(exist_ok=True, parents=True)

        self.path_log = self.dir_results / f"training_log_{time.strftime('%Y%m%d_%H%M%S')}.log"
        self.path_loss_fig = self.dir_results / f"loss_curve_{time.strftime('%Y%m%d_%H%M%S')}.png"
        self.dir_tensorboard_log = self.dir_results / "tensorboard_logs"

        self.writer = SummaryWriter(log_dir=self.dir_tensorboard_log)

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

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1} Training", leave=False):
            in_field = batch["in_field"].unsqueeze(1).to(self.device)
            mask = batch["mask"].unsqueeze(1).to(self.device)
            inputs = torch.cat([in_field, mask], dim=1)
            targets = batch["out_field"].unsqueeze(1).to(self.device)

            self.optimizer.zero_grad()

            # Forward pass under autocast for amp :
            with autocast(self.device):
                outputs = self.model(inputs)
                loss = self.compute_masked_mse_loss(outputs, targets, mask)

            # Backpropagation using GradScaler for amp :
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        msg = f"Epoch [{epoch + 1}/{self.num_epochs}] Training Loss: {avg_loss:.4f}"
        print(msg)
        logging.info(msg)

        # clear cache
        self.clear_cache()

        return avg_loss

    def validate_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                in_field = batch["in_field"].unsqueeze(1).to(self.device)
                mask = batch["mask"].unsqueeze(1).to(self.device)
                inputs = torch.cat([in_field, mask], dim=1)

                targets = batch["out_field"].unsqueeze(1).to(self.device)

                with autocast(self.device):
                    outputs = self.model(inputs)
                    loss = self.compute_masked_mse_loss(outputs, targets, mask)

                running_loss += loss.item()

            avg_loss = running_loss / len(self.val_loader)
            msg = f"Epoch [{epoch + 1}/{self.num_epochs}] Validation Loss: {avg_loss:.4f}"
            print(msg)
            logging.info(msg)

            # clear cache
            self.clear_cache()

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

    def clear_cache(self):
        import gc
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

    def log_mem_usage(self, epoch):

        if self.device == 'cuda':
            mem_alloc = torch.cuda.max_memory_allocated(self.device) / 1e6  # in MB
            mem_cached = torch.cuda.max_memory_reserved(self.device) / 1e6  # in MB
            logging.info(
                f"Epoch {epoch + 1}: Max GPU Memory Allocated: {mem_alloc:.2f} MB, "
                f"Max GPU Memory Reserved: {mem_cached:.2f} MB"
            )
            # Reset the peak stats for the next epoch.
            torch.cuda.reset_peak_memory_stats(self.device)

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

            # Append the losses for plotting
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # log GPU memory usage
            self.log_mem_usage(epoch)

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

            # Log the losses to tensorboard
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Validation", val_loss, epoch)

            for i, param_group in enumerate(self.optimizer.param_groups):
                self.writer.add_scalar(f"LR/group_{i}", param_group['lr'], epoch)

        self.writer.close()
