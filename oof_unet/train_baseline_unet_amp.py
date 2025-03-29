from monai.networks.nets.unet import UNet
import pandas as pd
import numpy as np
import argparse
import logging
import random
import torch
import time

from oof_unet.data_utils.datasets import AmpSingleCodeNiftiDataset
from oof_unet.data_utils.samplers import PreComputedVolumeBasedBatchSampler
from oof_unet.trainers import AmpOOFTrainer
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
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_lr", type=float, default=1e-3, help="Max Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
    parser.add_argument("--n_units", type=int, default=8, help="Number of units in each layer")
    parser.add_argument("--unit_volume", type=int, default=10_000_000, help="Volume of a single unit")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for the DataLoader")
    parser.add_argument("--oversampling_factor", type=int, default=10, help="Oversampling factor")
    parser.add_argument("--batches_per_epoch", type=int, default=200, help="Number of batches to train on")
    parser.add_argument("--prefetch_factor", type=int, default=4, help="Prefetch factor")
    parser.add_argument("--accumulation_steps", type=int, default=5, help="Accumulation steps")

    return parser.parse_args()


def main():

    args = parse_args()

    # Load the dataset :
    df_dataset = pd.read_csv("../csv/Dataset001_final_train.csv")

    logger.info(f"Amp Training ...")
    logger.info(f"Starting training with configuration : {vars(args)}")
    logger.info(f"Training for fold {args.fold} for {args.epochs} epochs : ")

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
    fold_name = f"Val_{args.fold}"

    list_train = df_dataset[df_dataset["fold"] != fold_name]["id_dosemap"].tolist()
    logger.info(f"Number of training samples : {len(list_train)}")
    train_dataset = AmpSingleCodeNiftiDataset(dir_data=constants.DIR_DATA,
                                              list_dosemaps=list_train,
                                              split="Train")

    train_sampler = PreComputedVolumeBasedBatchSampler(dataset=train_dataset,
                                                       n_units=args.n_units,
                                                       unit_volume=args.unit_volume,
                                                       oversampling_factor=args.oversampling_factor,
                                                       batches_per_epoch=args.batches_per_epoch)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_sampler=train_sampler,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               persistent_workers=True,
                                               prefetch_factor=args.prefetch_factor)

    list_val = df_dataset[df_dataset["fold"] == fold_name]["id_dosemap"].tolist()
    print(f"Number of validation samples : {len(list_val)}")
    val_dataset = AmpSingleCodeNiftiDataset(dir_data=constants.DIR_DATA,
                                            list_dosemaps=list_val,
                                            split="Train")

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             shuffle=False,
                                             batch_size=1,
                                             num_workers=args.num_workers,
                                             pin_memory=True,
                                            )

    # Set up the optimizer, criterion and scheduler :
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = MaskedMSELoss()
    scheduler = torch.optim.lr_scheduler. OneCycleLR(optimizer,
                                                     max_lr=args.max_lr,
                                                     steps_per_epoch=len(train_loader),
                                                     epochs=args.epochs,
                                                     pct_start=0.3,           # % of cycle spent increasing LR
                                                     anneal_strategy='cos',   # or 'linear'
                                                     div_factor=25.0,         # initial LR = max_lr / div_factor
                                                     final_div_factor=1e4     # final LR = max_lr / final_div_factor
                                                     )

    # Create the Trainer :
    trainer = AmpOOFTrainer(
        model=model,
        device=args.device,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        optimizer=optimizer,
        criterion=criterion,
        dir_results=constants.DIR_RESULTS / f"fold_{args.fold}",
        project_name="oof-unet",
        run_name=f"fold_{args.fold}",
        scheduler=scheduler,
        accumulation_steps=args.accumulation_steps
    )

    # Start training :
    logger.info("Training started ...")
    start_time = time.time()
    trainer.train()
    logger.info(f"Training completed in {time.time() - start_time} seconds !")


if __name__ == "__main__":
    main()
