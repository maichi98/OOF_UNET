from torch.utils.data import Dataset
from oof_unet import constants
import torch
import ants


class OofUniqueMachineDataset(Dataset):

    def __init__(self, list_dosemaps: list, split: str):
        self.list_dosemaps = list_dosemaps

    def __len__(self):
        return len(self.list_dosemaps)

    def __getitem__(self, idx):

        id_dosemap = self.list_dosemaps[idx]
        path_dosemap = constants.DIR_DATA / split / f"{id_dosemap}.nii.gz"

        # The dosemap volume is in the id_dosemap :
        volume = int(id_dosemap.split("_")[-1])

        # Read the nifti dose file
        dose = ants.image_read(str(path_dosemap))

        mask = (dose >= 0)
        dose = dose * mask

        thresh = dose.max() * 0.05

        in_field = (dose > thresh) * dose / 100
        out_field = (dose <= thresh) * dose / 100

        return {
            "in_field": torch.tensor(in_field).float(),
            "out_field": torch.tensor(out_field).float(),
            "mask": torch.tensor(mask).float(),
            "volume": volume
        }
