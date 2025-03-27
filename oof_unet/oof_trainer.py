import torch
import torch.optim as optim
import logging
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import random
import gc

from oof_unet import constants

# Set the random seeds for reproducibility :
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class OofTrainer:

    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 fold,
                 num_epochs=100,
                 lr=0.01,
                 device='cuda'):

        self.device = device
        if self.device == 'cuda':
            torch.cuda.manual_seed_all(42)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs

        # Set up optimizer, gradient scaler, and scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scaler = GradScaler()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

        self.dir_results = constants.DIR_RESULTS / f"fold_{fold}"
        self.dir_results.mkdir(exist_ok=True, parents=True)

        self.path_log = self.dir_results / f"training_log_{time.strftime('%Y%m%d_%H%M%S')}.log"
        self.path_loss_fig = self.dir_results / f"loss_curve_{time.strftime('%Y%m%d_%H%M%S')}.png"
        self.dir_tensorboard_log = self.dir_results / "tensorboard_logs"

        self.writer = SummaryWriter(log_dir=self.dir_tensorboard_log)

        # Set up the logger.
        logging.basicConfig(filename=self.path_log,
                            level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s')

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
        mask_sum = mask.sum() + epsilon
        loss = masked_error.sum() / mask_sum
        return torch.sqrt(loss)

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

    def save_checkpoint(self, epoch, tag):

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }

        if tag == 'best':
            filename = "best_checkpoint.pth"
        elif tag == 'every50':
            filename = f"checkpoint_epoch_{epoch + 1}.pth"
        else:
            filename = "last_checkpoint.pth"

        path = self.dir_results / filename
        torch.save(checkpoint, path)
        logging.info(f"Saved {tag} checkpoint at epoch {epoch + 1} to {path}")

    def train(self):
        for epoch in range(self.num_epochs):

            start_time = time.time()

            # Log the current learning rate before starting the epoch.
            current_lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
            logging.info(f"Epoch {epoch + 1}: Starting with learning rate(s): {current_lrs}")

            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch)

            logging.info(f"Epooch {epoch + 1} completed in {time.time() - start_time:.2f} seconds")

            # Append the losses for plotting and log GPU memory usage, etc.
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Log the GPU memory usage :
            self.log_mem_usage(epoch)

            # Save the model checkpoint at the end of each epoch :
            self.save_checkpoint(epoch, tag="last")

            # Save the best checkpoint if the validation loss has improved :
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, tag="best")
                logging.info(f"New best validation loss {val_loss:.4f} at epoch {epoch + 1}")

            # Save a checkpoint every 50 epochs :
            if (epoch + 1) % 50 == 0:
                self.save_checkpoint(epoch, tag="every50")

            # Plot the training and validation losses :
            self.plot_losses(epoch)

            # Log the losses to tensorboard.
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Validation", val_loss, epoch)

            # Log the current learning rate for each parameter group to tensorboard :
            for i, param_group in enumerate(self.optimizer.param_groups):
                self.writer.add_scalar(f"LR/group_{i}", param_group['lr'], epoch)

            # Step the learning rate scheduler with the training loss.
            self.scheduler.step(train_loss)
            updated_lrs = self.scheduler.get_last_lr()
            logging.info(f"Epoch {epoch + 1}: Updated learning rate after scheduler step: {updated_lrs}")

        self.writer.close()
