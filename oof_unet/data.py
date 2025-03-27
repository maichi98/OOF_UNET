from torch.utils.data import Dataset
from torch.utils.data import Sampler
from oof_unet import constants
import numpy as np
import random
import torch
import ants

random.seed(123)


class OofUniqueMachineDataset(Dataset):

    def __init__(self, list_dosemaps: list, split: str):
        self.list_dosemaps = list_dosemaps
        self.split = split

    def __len__(self):
        return len(self.list_dosemaps)

    def __getitem__(self, idx):

        id_dosemap = self.list_dosemaps[idx]
        path_dosemap = constants.DIR_DATA / self.split / f"{id_dosemap}.nii.gz"

        # The dosemap volume is in the id_dosemap :
        volume = int(id_dosemap.split("_")[-1])

        # Read the nifti dose file
        dose = ants.image_read(str(path_dosemap))

        mask = (dose >= 0)
        dose = dose * mask

        thresh = dose.max() * 0.05

        in_field = (dose > thresh) * dose / 100
        out_field = (dose <= thresh) * dose / 5

        return {
            "in_field": torch.tensor(in_field.numpy()).float(),
            "out_field": torch.tensor(out_field.numpy()).float(),
            "mask": torch.tensor(mask.numpy()).float(),
            "volume": volume
        }


class VolumeBasedBatchSampler(Sampler):

    def __init__(self, dataset, n_units: int, unit_volume: int = 10_000_000):

        super().__init__()

        self.dataset = dataset
        self.max_voxels = n_units * unit_volume
        self.dict_categories = self._group_by_volume()

    def _group_by_volume(self):

        list_dosemaps = self.dataset.list_dosemaps

        dict_categories = {}

        for i, id_dosemap in enumerate(list_dosemaps):

            volume = int(id_dosemap.split("_")[-1])

            if volume not in dict_categories:
                dict_categories[volume] = []

            dict_categories[volume].append(i)

        return dict_categories

    def __iter__(self):

        # Shuffle the categories
        categories = list(self.dict_categories.keys())
        random.shuffle(categories)

        for volume in categories:
            # Shuffle the order of the dosemaps in the category
            indices = self.dict_categories[volume]
            random.shuffle(indices)

            # Compute the desired batch size
            batch_size = max(1, self.max_voxels // volume)

            # Yield the batches:
            for i in range(0, len(indices), batch_size):
                yield indices[i:i + batch_size]

    def __len__(self):

        total_batches = 0

        for volume, category_dosemaps in self.dict_categories.items():

            batch_size = max(1, self.max_voxels // volume)
            total_batches += int(np.ceil(len(category_dosemaps) / batch_size))

        return int(total_batches)
