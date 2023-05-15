import torch
from torch.utils.data import Dataset
from astropy.io import fits
from pathlib import Path
import numpy as np
from torchvision.transforms import RandomCrop


import torch
from torch.utils.data import Dataset
from astropy.io import fits
from pathlib import Path
import numpy as np
from torchvision.transforms import RandomCrop
from typing import Tuple


class FitsDataset(Dataset):
    """
    Dataset class for loading FITS files with optional random cropping to allow inconsistent sized files.

    Args:
        root_dir (str): Root directory containing the FITS files.
        hdu_index (int): Index of the HDU to extract data from. Default is 0 (primary HDU).
        num_channels (int): Number of channels for the image. Default is 1.
        data_type (torch.dtype): Desired data type of the tensor. Default is torch.float32.
        crop_size (Tuple[int, int]): Size of the random crop. Default is (256, 256).

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
        num_channels: int = 1,
        data_type: torch.dtype = torch.float32,
        crop_size: Tuple[int, int] = (256, 256),
    ):
        self.file_paths = self._get_file_paths(root_dir)
        self.hdu_index = hdu_index
        self.num_channels = num_channels
        self.data_type = data_type
        self.crop_transform = RandomCrop(crop_size)

    def _get_file_paths(self, root_dir: str) -> list:
        root_path = Path(root_dir)
        file_paths = list(root_path.glob("**/*.fits"))
        return file_paths

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        file_path = self.file_paths[index]
        with fits.open(file_path) as hdul:
            data = hdul[self.hdu_index].data.astype("float32")

        if self.num_new_channels is not None:
            data = np.expand_dims(data, axis=0)  # Add channel dimension
            data = np.repeat(
                data, self.num_new_channels, axis=0
            )  # Repeat data along the channel dimension

        data_tensor = torch.tensor(data, dtype=self.data_type)
        data_tensor = self.crop_transform(data_tensor)

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
