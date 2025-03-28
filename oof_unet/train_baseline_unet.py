from monai.networks.nets.unet import UNet
import pandas as pd
import numpy as np
import argparse
import logging
import random
import torch
import time

from oof_unet.data_utils.datasets import SingleCodeNiftiDataset
from oof_unet.data_utils.samplers import VolumeBasedBatchSampler
from oof_unet.trainers.base_trainer import BaseOOFTrainer
from oof_unet.loss_functions import MaskedMSELoss
from oof_unet import constants

# Set the random seeds for reproducibility :
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


# Setup logging :
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set up the formatter :
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=250, help="Number of training epochs")
    parser.add_argument("--fold", type=int, required=True, help="Fold number to train on")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
    parser.add_argument("--n_units", type=int, default=6, help="Number of units in each layer")
    parser.add_argument("--unit_volume", type=int, default=10_000_000, help="Volume of a single unit")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for the DataLoader")

    return parser.parse_args()


def main():

    args = parse_args()

    # Load the dataset :
    df_dataset = pd.read_csv("../csv/Dataset001_final_train.csv")

    logger.info(f"Starting training with configuration : {vars(args)}")
    logger.info(f"Training for fold {args.fold} for {args.epochs} epochs : ")

    # Set up the model :
    def group_norm(num_channels):
        return torch.nn.GroupNorm(num_groups=8, num_channels=num_channels)

    model = UNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="INSTANCE"
    )

    # Create the train_loader and the val_loader :
    list_train = df_dataset[df_dataset["fold"] != args.fold]["id_dosemap"].tolist()
    train_dataset = SingleCodeNiftiDataset(dir_data=constants.DIR_DATA, list_dosemaps=list_train, split="Train")
    train_sampler = VolumeBasedBatchSampler(dataset=train_dataset, n_units=args.n_units, unit_volume=args.unit_volume)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=args.num_workers)

    list_val = df_dataset[df_dataset["fold"] == args.fold]["id_dosemap"].tolist()
    val_dataset = SingleCodeNiftiDataset(dir_data=constants.DIR_DATA, list_dosemaps=list_val, split="Train")
    val_sampler = VolumeBasedBatchSampler(dataset=val_dataset, n_units=args.n_units, unit_volume=args.unit_volume)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=args.num_workers)

    # Set up the optimizer, criterion and scheduler :
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = MaskedMSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Create the Trainer :
    trainer = BaseOOFTrainer(
        model=model,
        device=args.device,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        optimizer=optimizer,
        criterion=criterion,
        dir_results=constants.DIR_RESULTS,
        project_name="oof-unet",
        run_name=f"fold_{args.fold}",
        scheduler=scheduler
    )

    # Start training :
    logger.info("Training started ...")
    start_time = time.time()
    trainer.train()
    logger.info(f"Training completed in {time.time() - start_time} seconds !")


if __name__ == "__main__":
    main()
