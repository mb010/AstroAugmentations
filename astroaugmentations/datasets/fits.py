import random
import torch
from torch.utils.data import Dataset
from astropy.io import fits
from pathlib import Path
import numpy as np
from torchvision.transforms import RandomCrop
import albumentations as A

import torch
from torch.utils.data import Dataset
from astropy.io import fits
from pathlib import Path
import numpy as np
from torchvision.transforms import RandomCrop
from typing import Tuple, Optional


class FitsDataset(Dataset):
    """
    Dataset class for loading FITS files with optional random cropping to allow inconsistent sized files.

    Args:
        root_dir (str): Root directory containing the FITS files.
        hdu_index (int): Index of the HDU to extract data from. Default is 0 (primary HDU).
        num_channels (int): Number of channels for the image. Default is 1.
        data_type (torch.dtype): Desired data type of the tensor. Default is torch.float32.
        crop_size (Tuple[int, int]): Size of the random crop. Default is (256, 256).
        transform (Optional[callable]): transform to apply to the images.

    Returns:
        Tuple[torch.Tensor, str]: A tuple containing the data tensor and the file name.

    Example:
        root_directory = '/path/to/data/folder'
        hdu_index = 1
        num_channels = 3
        data_type = torch.bfloat16
        crop_size = (256, 256)
        dataset = FitsDataset(root_directory, hdu_index, num_channels, data_type, crop_size)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        for batch, file_names in data_loader:
            # Use the batch of data and file names here
            print(batch.shape)
            print(batch.dtype)
            print(file_names)
    """

    def __init__(
        self,
        root_dir: str,
        hdu_index: int = 0,
        num_new_channels: int = None,
        data_type: torch.dtype = torch.float32,
        crop_size: Tuple[int, int] = (256, 256),
        stage: Optional[str] = None,
        seed: Optional[int] = None,
        transform: Optional[callable] = None,
        aug_type: Optional[str] = None,
        test_fraction: float = 0.15,
        val_fraction: float = 0.15,
        memmap: bool = True,
        pre_load: bool = False,
    ):
        self.hdu_index = hdu_index
        self.num_new_channels = num_new_channels
        self.data_type = data_type
        self.test_fraction = test_fraction
        self.val_fraction = val_fraction
        self.crop_transform = RandomCrop(crop_size)
        self.transform = transform
        self.aug_type = aug_type
        self.file_paths = self._get_file_paths(root_dir)  # get paths
        random.Random(seed).shuffle(self.file_paths)  # shuffle in place
        # split according to stage
        self.file_paths = self._split_data(stage)
        self.memmap = memmap
        self.pre_load = pre_load
        if self.pre_load:
            self.images = self._load_images()

    def _split_data(self, stage):
        # Stage must be one of ['fit', 'validate', 'test', 'predict']
        val_index = int(self.__len__() * (1 - self.val_fraction - self.test_fraction))
        test_index = int(self.__len__() * (1 - self.test_fraction))
        if stage == "fit":
            return self.file_paths[:val_index]
        elif stage == "validate":
            return self.file_paths[val_index:test_index]
        elif stage == "test":
            return self.file_paths[test_index:]
        else:
            return self.file_paths

    def _get_file_paths(self, root_dir: str) -> list:
        root_path = Path(root_dir)
        file_paths = list(root_path.glob("**/*.fits"))
        return file_paths

    def _load_images(self) -> list:
        images = []
        for file in self.file_paths:
            with fits.open(file, memmap=self.memmap) as hdul:
                img = hdul[self.hdu_index].data.astype(np.float32)
            images.append(img)
        return images

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        if self.pre_load:
            image = self.images[index]
        else:
            file_path = self.file_paths[index]
            with fits.open(file_path, memmap=self.memmap) as hdul:
                data = hdul[self.hdu_index].data.astype(np.float32)

        if self.num_new_channels is not None:
            data = np.expand_dims(data, axis=0)  # Add channel dimension
            data = np.repeat(
                data, self.num_new_channels, axis=0
            )  # Repeat data along the channel dimension

        data_tensor = torch.tensor(data, dtype=self.data_type)
        data_tensor = self.crop_transform(data_tensor)

        data_tensor = data_tensor.squeeze()
        if self.transform is not None:
            if self.aug_type == "albumentations":
                data_tensor = self.transform(image=np.asarray(data_tensor))["image"]
                data_tensor = torch.from_numpy(data_tensor).to(self.data_type)
            else:
                data_tensor = self.transform(data_tensor).to(self.data_type)

        file_name = file_path.name  # Extract file name
        return data_tensor, file_name  # Return data and file name

    def __len__(self) -> int:
        return len(self.file_paths)


def main():
    # Usage example
    root_directory = "/path/to/data/folder"
    hdu_index = 1  # Specify the HDU index (0 for primary HDU)
    num_new_channels = 3  # Specify the number of channels for the image
    data_type = torch.bfloat16  # Specify the desired data type
    dataset = FitsDataset(root_directory, hdu_index, num_new_channels, data_type)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Accessing data using the data loader
    for batch, file_names in data_loader:
        # Use the batch of data and file names here
        print(
            batch.shape
        )  # Example: (32, 3, N, M), where N and M are dimensions of each FITS image
        print(batch.dtype)  # Example: torch.bfloat16
        print(file_names)  # List of file names in the current batch


if __name__ == "__main__":
    main()
