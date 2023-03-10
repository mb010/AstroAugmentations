# 05.07.2022

import tensorflow as tf
import h5py
import numpy as np

import tensorflow_datasets as tfds


class GalaxyMNIST(tfds.core.GeneratorBasedBuilder):
    """`GalaxyMNIST <https://github.com/mwalmsley/galaxy_mnist>`_ Dataset.

    Based on MNIST/FashionMNIST torchvision datasets.

    Args:
        data_dir (string): Root directory of dataset where ``GalaxyMNIST/raw/train_dataset.hdf5``
            and  ``GalaxyMNIST/raw/test_dataset.hdf5`` exist.
        split (bool, optional): If 'train', creates dataset from ``GalaxyMNIST/raw/train_dataset.hdf5``,
            otherwise from ``GalaxyMNIST/raw/test_dataset.hdf5``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    
    VERSION = tfds.core.Version('0.1.0')
    RELEASE_NOTES = { '0.1.0': 'First Test',}
    
    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation, ...)."""
        return tfds.core.DatasetInfo(
            builder=self,
            description="""
                Galaxy Zoo DECaLS, GalaxyMNIST dataset by M. Walmsley and DECaLS collaboration
                """,
            homepage="https://github.com/mwalmsley/galaxy_mnist",
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(shape=(64,64,3)),
                'label': tfds.features.ClassLabel(
                    names=["smooth_round", "smooth_cigar", "edge_on_disk", "unbarred_spiral"])
            }),
            citation=r"""
            @ARTICLE{2022MNRAS.509.3966W,
            author = {{Walmsley}, Mike and {Lintott}, Chris and {G{\'e}ron}, Tobias and {Kruk}, Sandor and {Krawczyk}, Coleman and {Willett}, Kyle W. and {Bamford}, Steven and {Kelvin}, Lee S. and {Fortson}, Lucy and {Gal}, Yarin and {Keel}, William and {Masters}, Karen L. and {Mehta}, Vihang and {Simmons}, Brooke D. and {Smethurst}, Rebecca and {Smith}, Lewis and {Baeten}, Elisabeth M. and {Macmillan}, Christine},
            title = "{Galaxy Zoo DECaLS: Detailed visual morphology measurements from volunteers and deep learning for 314 000 galaxies}",
            journal = {\mnras},
            keywords = {methods: data analysis, galaxies: bar, galaxies: general, galaxies: interactions, Astrophysics - Astrophysics of Galaxies, Computer Science - Computer Vision and Pattern Recognition},
             year = 2022,
            month = jan,
            volume = {509},
            number = {3},
            pages = {3966-3988},
              doi = {10.1093/mnras/stab2093},
            archivePrefix = {arXiv},
            eprint = {2102.08414},
            primaryClass = {astro-ph.GA},
            adsurl = {https://ui.adsabs.harvard.edu/abs/2022MNRAS.509.3966W},
            adsnote = {Provided by the SAO/NASA Astrophysics Data System}
            }
            """,
        )
    
    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Download the data and define splits"""
        extracted_path = dl_manager.download_and_extract({
            'train': "http://www.jb.man.ac.uk/research/MiraBest/MiraBest_F/train_dataset.hdf5.gz",
            'test': "http://www.jb.man.ac.uk/research/MiraBest/MiraBest_F/test_dataset.hdf5.gz"
        #    'train':"/home/pearsonw/dev/deep_learning/AstroAugmentations/data/GalaxyMNIST/raw/train_dataset.hdf5.gz",
        #    'test':"/home/pearsonw/dev/deep_learning/AstroAugmentations/data/GalaxyMNIST/raw/test_dataset.hdf5.gz"
        })
        return {
            'train': self._generate_examples(path=extracted_path['train']),
            'test': self._generate_examples(path=extracted_path ['test']),
        }
    
    def _generate_examples(self, path):
        """Geneartor of examples for each split"""
        print('generator')
        with h5py.File(path, 'r') as f:
            images = f['images'][:]
            images = images.astype(np.uint8)
            # images are saved as NHWC convention

            targets = f['labels'][:]  # integer-encoded (from 0) according to GalaxyMNIST.classes
            targets = targets.astype(np.int64) # dtype consistent with mnist (same as tensor.long())
            
            f.close()
            
            for i in range(0, len(images)):
                yield i, {'image':images[i], 'label':targets[i]}