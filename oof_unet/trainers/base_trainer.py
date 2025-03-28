import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import torch
import wandb
import time


class BaseOOFTrainer:

    def __init__(self,
                 model,
                 device,
                 train_loader,
                 val_loader,
                 num_epochs,
                 optimizer,
                 criterion,
                 dir_results,
                 project_name,
                 run_name,
                 scheduler=None):

        self.device = device
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.num_epochs = num_epochs

        self.optimizer = optimizer
        self.criterion = criterion

        self.dir_results = dir_results
        self.scheduler = scheduler

        # Initialize wandb :
        wandb.init(project=project_name,
                   name=run_name,
                   config={
                       "model": model.__class__.__name__,
                       "num_epochs": num_epochs,
                       "optimizer": optimizer.__class__.__name__,
                       "criterion": criterion.__class__.__name__
                   })
        # Automatically log gradients and network topology :
        wandb.watch(self.model, log="all", log_freq=100)

        self.best_val_loss = float('inf')

    def train_epoch(self, epoch):

        start_time = time.time()

        # Set the model to training mode :
        self.model.train()

        # Initialize the loss value for the epoch :
        epoch_loss = 0.0

        # Iterate over the training data :
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1} Training", leave=False):

            in_field = batch["in_field"].unsqueeze(1).to(self.device)
            mask = batch["mask"].unsqueeze(1).to(self.device)
            inputs = torch.cat([in_field, mask], dim=1)

            targets = batch["out_field"].unsqueeze(1).to(self.device)

            # Zero the gradients :
            self.optimizer.zero_grad()

            # Forward pass :
            outputs = self.model(inputs)
            loss = self.criterion(**{"outputs": outputs, "targets": targets, "mask": mask})

            # Backpropagation :
            loss.backward()
            # Apply gradient clipping :
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Accumulate the loss value :
            epoch_loss += loss.item()

        return epoch_loss, time.time() - start_time

    def validate_epoch(self, epoch):

        # Set the model to evaluation mode :
        self.model.eval()

        # Initialize the loss value for the epoch :
        epoch_loss = 0.0

        # Iterate over the validation data :
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {epoch + 1} Validation", leave=False):

                in_field = batch["in_field"].unsqueeze(1).to(self.device)
                mask = batch["mask"].unsqueeze(1).to(self.device)
                inputs = torch.cat([in_field, mask], dim=1)

                targets = batch["out_field"].unsqueeze(1).to(self.device)

                # Forward pass :
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Accumulate the loss value :
                epoch_loss += loss.item()

        return epoch_loss

    def update_lr(self, val_loss=None):

        if self.scheduler is None:
            return

        if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)

        else:
            self.scheduler.step()

    def save_checkpoint(self, epoch, tag):

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        dict_filenames = {"best": "best_checkpoint.pth",
                          "last": "last_checkpoint.pth"}
        torch.save(checkpoint, self.dir_results / dict_filenames[tag])

    def train(self):

        for epoch in range(self.num_epochs):

            train_loss, epoch_duration = self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch)
            current_lr = self.optimizer.param_groups[0]["lr"]
            gradient_norm = self._compute_gradient_norm()

            print(f"Epoch [{epoch + 1}/{self.num_epochs}] Training Loss: {train_loss:.4f} "
                  f"Validation Loss: {val_loss:.4f} | LR: {current_lr:.6f} | Duration: {epoch_duration:.2f}s")

            # Log metrics to wandb :
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "gradient_norm": gradient_norm
            })

            # Save the model checkpoints:
            self.save_checkpoint(epoch, "last")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, "best")

            # Update the learning rate :
            self.update_lr(val_loss)

        print("Training completed successfully !")
        wandb.finish()

    def _compute_gradient_norm(self):

        total_norm = 0.0
        for p in self.model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

