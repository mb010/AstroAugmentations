# AstroAugmentations
Custom image augmentations specifically designed around astronomical
instruments and data. Please open an
[issue](https://github.com/mb010/AstroAugmentations/issues) to highlight missing augmentations and / or datasets. This is an open source project, so feel free to fork, make changes and submit a [pull request](https://github.com/mb010/AstroAugmentations/pulls) of your additions and modifications!

This package is based on [Albumentations](https://github.com/albumentations-team/albumentations/).
This should allow scalability and applicability in a multitude of cases,
including both TensorFlow and PyTorch.

# Features
- Augmentations designed around specific astronomical domains and data formats
- Access to standardized default data sets.
- This early version (only) covers:
  - Radio images (designed with interferometers in mind)

# Installation
**Install**: `pip install AstroAugmentations`\
**Suggested Import**: `import astroaugmentations as AA`.

:warning: **Import will fail in v0.1.0 if torch and torchvision are not installed!**
Which version you install depends on your system.
Please see the official [PyTorch](https://pytorch.org/) site to download
an appropriate configuration.
(Developed using: `torch>=1.10.2+cu113` and `'torchvision>=0.11.3+cu113`).
These are currently used in the example datasets.

# Usage
Example augmentations for all modalities and domains supported are provided within the `AA.AstroAugmentations()` class in [this file](https://github.com/mb010/AstroAugmentations/tree/main/astroaugmentations/augmentations.py).

`AA.CustomKernelConvolution()` requires a kernel to be available in a directory as a saved numpy array (e.g. `./kernels/FIRST_kernel.npy`). We provide a kernel we generated for the
[FIRST Survey](http://sundog.stsci.edu/)
[here](https://github.com/mb010/AstroAugmentations/tree/main/astroaugmentations/kernels).

## Data Sets
Data sets are called using the scripts provided in
[astroaugmentations/datasets](https://github.com/mb010/AstroAugmentations/tree/main/astroaugmentations/datasets).
Please see the `*.ipynb` notebooks for demonstrations on how best to use these.

# Demo / Examples
Please see the ipython notebooks provided for demonstrations of the
various augmentations. These are implemented using Torch.
The interaction with the Albumentations package should allow for
AstroAugmentations to be applied to other frameworks.
See examples of their implementations [here](https://albumentations.ai/docs/examples/).

# Package Structure:
```
AstroAugmentations
├── LICENSE
├── astroaugmentations
│   ├── __init__.py
│   ├── augmentations.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── VLA_raw_antenna_position.py
│   │   └── kernel_creation.py
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── galaxy_mnist.py
│   │   └── MiraBest_F.py
│   └── module_numpy_2.py
├── README.md
└── setup.py
```

# Citation
Relevant publication in prep. Please reach out to the author for details:\
micah.bowles@postgrad.manchester.ac.uk
