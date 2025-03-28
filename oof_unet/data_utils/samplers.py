from torch.utils.data import Sampler
import numpy as np
import random
random.seed(42)


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
