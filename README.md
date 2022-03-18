# AstroAugmentations
Custom image augmentations specifically designed around astronomical
instruments and data. Please open an
[issue](https://github.com/mb010/AstroAugmentations/issues) to highlight missing augmentations and / or datasets. This is an open source project, so feel free to fork, make changes and submit a [pull request](https://github.com/mb010/AstroAugmentations/pulls) of your additions and modifications!

This package is in active development and although it should work, it may be a bit temporamental and require some love to get to know. Feel free to make suggestions using the [issue](https://github.com/mb010/AstroAugmentations/issues) tracker.

This package is based on [Albumentations](https://github.com/albumentations-team/albumentations/).
This should allow scalability and applicability in a multitude of cases,
including both TensorFlow and PyTorch.

# Features
- Augmentations designed for specific astronomical domains and data formats.
- Access to standardized default data sets.
- Most recent version covers:
  - Radio image augmentations (designed with interferometers in mind)

# Quick Start
**Importing**: `import astroaugmentations as AA`.

**Install**:
```python
pip install -U git+https://github.com/mb010/AstroAugmentations.git
pip install -U git+https://github.com/albumentations-team/albumentations
```
:warning: **Currently requires torch and torchvision which are not autmatically installed!**
The version you install depends on your system.
Please see the official [PyTorch](https://pytorch.org/) site to download
an appropriate configuration. These are currently used in the example datasets.\
Developed using: `torch>=1.10.2+cu113` and `'torchvision>=0.11.3+cu113`.

# Usage

The default is to import the package as `AA`: `import astroaugmentations as AA`.
Augmentations for each data type are seperated into individual modules, 
each of which will contain submodules with regime specific augmentations e.g.:
- `AA.image_domain` contains transformations designed for imaging / computer vision tasks.
  - `AA.image_domain.optical` provides augmentations specifically designed around [optical imaging](https://github.com/mb010/AstroAugmentations/tree/main/astroaugmentations/image_domain/optical.py).
  - `AA.image_domain.radio` provides augmentations specifically designed around [radio imaging](https://github.com/mb010/AstroAugmentations/tree/main/astroaugmentations/image_domain/radio.py).

`AA.composed` contains 'ready to go' 
[example compositions](https://github.com/mb010/AstroAugmentations/blob/main/astroaugmentations/composed.py) 
of multiple transforms explicitly designed for a data type and regime.

`AA.CustomKernelConvolution()` requires a kernel to be available in a directory as 
a saved numpy array (e.g. `./kernels/FIRST_kernel.npy`). We provide a kernel we generated
[here](https://github.com/mb010/AstroAugmentations/tree/main/astroaugmentations/kernels)
(designed for the [FIRST Survey](http://sundog.stsci.edu/)).

# Demo / Examples
Please see the ipython notebooks provided for demonstrations of the
various augmentations. These are implemented using Torch.
The interaction with the Albumentations package should allow for
AstroAugmentations to be applied to other frameworks.
See examples of their implementations [here](https://albumentations.ai/docs/examples/).

# Using the in-built datasets
Data sets are provided in
[astroaugmentations/datasets](https://github.com/mb010/AstroAugmentations/tree/main/astroaugmentations/datasets).
See use examples in the demonstration ipython notebooks.

# Adapting Data Loaders (PyTorch)
Following Albumentions notation, we adapt respective torch data loaders from a functional call to an Albumnetations call as shown in their [PyTorch Example](https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/#Define-a-PyTorch-dataset-class) which allows respective transformations to be applied simultaneously to segmentation masks. We present an example of what this can look like.

Assuming there is a `self.transform` attribute as a parameter in our data class. In which case, normally inside the `__getitem__` method, a conditional application of the transform is made:
```
if self.transform is not None:
    image = self.transform(image)
```
For Albumentations, and thus our package, we need to adapt this notation. In the case of image augmentations (no mask augmentations) we write:
```
if self.transform is not None:
    image = self.transform(image=image)["image"]
```
This seems unnecessary, until we consider an example of what happens when we try to apply our transformations to masks as well as the input:
```
if self.transform is not None:
    transformed = self.transform(image=image, mask=mask)
    image = transformed["image"]
    mask = transformed["mask"]
```


# Package Structure:
```
AstroAugmentations
├── LICENSE
├── astroaugmentations
│   ├── __init__.py
│   ├── image_domain
│   │   ├── general.py
│   │   ├── optical.py
│   │   └── radio.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── VLA_raw_antenna_position.py
│   │   └── kernel_creation.py
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── galaxy_mnist.py
│   │   └── MiraBest_F.py
│   └── composed.py
├── README.md
└── setup.py
```

# Citation
Relevant publication in prep. Please reach out to the author for updates.

# Contact
For questions please contact: micah.bowles@postgrad.manchester.ac.uk \
For bugs or any issues with implementing this package, please open an [issue](https://github.com/mb010/AstroAugmentations/issues).
