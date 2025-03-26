import argparse
import pandas as pd

from oof_unet.unet import init_baseline_model
from oof_unet.data import OofUniqueMachineDataset, VolumeBasedBatchSampler
from oof_unet.trainer import OofTrainer
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet model on dosemap data")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--fold", type=int, required=True,
                        help="Fold number to train on")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--n_units", type=int, default=5,
                        help="Batch size for training")
    parser.add_argument("--unit_volume", type=int, default=10_000_000,
                        help="Volume of a single unit")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to train on")

    return parser.parse_args()


def main():

    args = parse_args()

    print(fr"Starting training for fold {args.fold} for {args.num_epochs} epochs : ")

    df_dataset = pd.read_csv('Dataset001_final_train.csv')

    # Create the model :
    model = init_baseline_model()

    # Create the train and val datasets :
    df_train = df_dataset[df_dataset['fold'] != f'Val_{args.fold}']
    df_val = df_dataset[df_dataset['fold'] == f'Val_{args.fold}']

    list_train = df_train['id_dosemap'].tolist()
    train_dataset = OofUniqueMachineDataset(list_dosemaps=list_train, split='Train')
    train_sampler = VolumeBasedBatchSampler(dataset=train_dataset, n_units=args.n_units, unit_volume=args.unit_volume)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4)

    list_val = df_val['id_dosemap'].tolist()
    val_dataset = OofUniqueMachineDataset(list_dosemaps=list_val, split='Train')
    val_sampler = VolumeBasedBatchSampler(dataset=val_dataset, n_units=args.n_units, unit_volume=args.unit_volume)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=4)

    # Create the trainer :
    trainer = OofTrainer(model=model,
                         train_loader=train_loader,
                         val_loader=val_loader,
                         fold=args.fold,
                         num_epochs=args.num_epochs,
                         lr=args.lr,
                         device=args.device)


