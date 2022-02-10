# AstroAugmentations
A package with various custom image augmentations implemented which are specifically designed around astronomical instruments and data sets.

# Quick Start
Install using `pip install astroaugmentations`.
Import `import AstroAugmentations as AA`.
Import datasets `from AstroAugmentations import datasets`.

### Note:
torch and torchvision are also required!
Which version you install depends on your system. Please see: https://pytorch.org/
We used: `torch>=1.10.2+cu113` and `'torchvision>=0.11.3+cu113`.

## Package Structure:
AstroAugmentations
├── LICENSE
├── src
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
