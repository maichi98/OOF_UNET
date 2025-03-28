from torch.utils.data import Dataset
from pathlib import Path
import torch
import ants


class SingleCodeNiftiDataset(Dataset):
    """
    Dataset class for loading single Machine code nifti Dosemaps
    """

    def __init__(self,
                 dir_data: Path,
                 list_dosemaps: list,
                 split: str):

        self.dir_data = dir_data
        self.list_dosemaps = list_dosemaps
        self.split = split

    def __len__(self):
        return len(self.list_dosemaps)

    def __getitem__(self, idx):

        id_dosemap = self.list_dosemaps[idx]
        path_dosemap = self.dir_data / self.split / f"{id_dosemap}.nii.gz"

        dose = ants.image_read(str(path_dosemap))

        mask = (dose >= 0)
        dose = dose * mask

        thresh = dose.max() * 0.05
        in_field = (dose > thresh) * dose / 100
        out_field = (dose <= thresh) * dose / 5

        return {
            "in_field": torch.tensor(in_field.numpy()).float(),
            "out_field": torch.tensor(out_field.numpy()).float(),
            "mask": torch.tensor(mask.numpy()).float()
        }
