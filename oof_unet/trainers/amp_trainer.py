import torch.optim.lr_scheduler as lr_scheduler
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import torch
import wandb
import time

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True



class AmpOOFTrainer:

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
                 scheduler=None,
                 accumulation_steps=5):

        self.device = device
        self.model = model.to(self.device)
        # Enable cuDNN autotuner to find the best algorithm for the current hardware and input size.
        # Works best when input sizes do not vary (e.g., fixed 3D volume shapes in UNet).
        torch.backends.cudnn.benchmark = True
        # try:
        #     self.model = torch.compile(model)
        # except Exception as e:
        #     print(fr"Failed to compile model: {e}")

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.num_epochs = num_epochs

        self.scaler = GradScaler(self.device)

        self.optimizer = optimizer
        self.criterion = criterion

        self.dir_results = dir_results
        self.dir_results.mkdir(parents=True, exist_ok=True)

        self.scheduler = scheduler

        self.accumulation_steps = accumulation_steps

        # Initialize wandb :
        wandb.init(project=project_name,
                   name=run_name,
                   config={
                       "model": model.__class__.__name__,
                       "num_epochs": num_epochs,
                       "optimizer": optimizer.__class__.__name__,
                       "criterion": criterion.__class__.__name__,
                       "scheduler": scheduler.__class__.__name__
                   })
        # Automatically log gradients and network topology :
        wandb.watch(self.model, log="all", log_freq=10)

        self.best_val_loss = float('inf')

        self.scheduler_batch_level = isinstance(self.scheduler, lr_scheduler.OneCycleLR)
        self.scheduler_requires_val_loss = isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau)

    def train_epoch_with_accumulation(self, epoch):

        start_time = time.time()
        self.model.train()

        epoch_loss = 0.0
        self.optimizer.zero_grad()

        for i, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch + 1} Training", leave=False)):
            batch: dict

            in_field = batch["in_field"].unsqueeze(1).to(dtype=torch.float32, device=self.device, non_blocking=True)
            mask = batch["mask"].unsqueeze(1).to(dtype=torch.float32, device=self.device, non_blocking=True)
            inputs = torch.cat([in_field, mask], dim=1)
            targets = batch["out_field"].unsqueeze(1).to(dtype=torch.float32, device=self.device, non_blocking=True)

            with autocast(self.device):
                outputs = self.model(inputs)
                outputs = torch.clamp(outputs, 0, 10)
                loss = self.criterion(outputs=outputs, targets=targets, mask=mask)

            # Normalize loss for accumulation
            loss = loss / self.accumulation_steps
            self.scaler.scale(loss).backward()

            # Step every N batches
            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self.scheduler_batch_level:
                    self.scheduler.step()

            # Accumulate the true loss for logging
            epoch_loss += loss.item() * self.accumulation_steps

        return epoch_loss, time.time() - start_time

    def validate_epoch(self, epoch):

        # Set the model to evaluation mode :
        self.model.eval()

        # Initialize the loss value for the epoch :
        epoch_loss = 0.0

        # Iterate over the validation data :
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {epoch + 1} Validation", leave=False):

                in_field = batch["in_field"].unsqueeze(1).to(dtype=torch.float32, device=self.device, non_blocking=True)
                mask = batch["mask"].unsqueeze(1).to(dtype=torch.float32, device=self.device, non_blocking=True)
                inputs = torch.cat([in_field, mask], dim=1)

                targets = batch["out_field"].unsqueeze(1).to(dtype=torch.float32, device=self.device, non_blocking=True)

                # Forward pass :
                with autocast(self.device):
                    outputs = self.model(inputs)
                    outputs = torch.clamp(outputs, 0, 10)
                    loss = self.criterion(outputs=outputs, targets=targets, mask=mask)

                # Accumulate the loss value :
                epoch_loss += loss.item()

        return epoch_loss

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

            train_loss, epoch_duration = self.train_epoch_with_accumulation(epoch)
            val_loss = self.validate_epoch(epoch)
            current_lr = self.optimizer.param_groups[0]["lr"]
            gradient_norm = self._compute_gradient_norm()

            print(f"Epoch [{epoch + 1}/{self.num_epochs}] Training Loss: {train_loss:.4f} "
                  f"Validation Loss: {val_loss:.4f} | LR: {current_lr:.6f} | Duration: {epoch_duration:.2f}s")

            # Log metrics to wandb :
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "gradient_norm": gradient_norm,
                "epoch": epoch,
                "epoch_duration": epoch_duration,
            })

            # Save the model checkpoints:
            self.save_checkpoint(epoch, "last")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, "best")

            # Update the learning rate :
            if not self.scheduler_batch_level:
                if self.scheduler_requires_val_loss:
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

        print("Training completed successfully !")
        wandb.finish()

    def _compute_gradient_norm(self):

        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5 if total_norm > 0 else 0.0
