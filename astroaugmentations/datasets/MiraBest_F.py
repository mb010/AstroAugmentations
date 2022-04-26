import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from sklearn.model_selection import train_test_split
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

from PIL import Image


class MiraBest_F(data.Dataset):
    """
    Inspired by `HTRU1 <https://as595.github.io/HTRU1/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``MiraBest-F.py` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        test_size (float, optional): Fraction of data to be stratified into a test set. i.e. 0.2
            stratifies 20% of the MiraBest into a test set. Default (None) returns the
            standard MiraBest data set.
    """

    base_folder = "F_batches"
    url = (
        "http://www.jb.man.ac.uk/research/MiraBest/MiraBest_F/MiraBest_F_batches.tar.gz"
    )
    filename = "MiraBest_F_batches.tar.gz"
    tgz_md5 = "7d4e3a623d29db7204bce81676ee8ce2"
    train_list = [
        ["data_batch_1", "f7a470b7367e8e0d0c5093d2cf266d54"],
        ["data_batch_2", "bb65ecd7e748e9fb789419b1efbf1bab"],
        ["data_batch_3", "32de1078e7cd47f5338c666a1b563ede"],
        ["data_batch_4", "a1209aceedd8806c88eab27ce45ee2c4"],
        ["data_batch_5", "1619cd7c54f5d71fcf4cfefea829728e"],
        ["data_batch_6", "636c2b84649286e19bcb0684fc9fbb01"],
        ["data_batch_7", "bc67bc37080dc4df880ffe9720d680a8"],
    ]

    test_list = [
        ["test_batch", "ac7ea0d5ee8c7ab49f257c9964796953"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "e1b5450577209e583bc43fbf8e851965",
    }

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        test_size=None,
        aug_type="albumentations",
    ):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.aug_type = aug_type

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        if self.train and test_size is None:
            downloaded_list = self.train_list
        elif not self.train and test_size is None:
            downloaded_list = self.test_list
        else:
            downloaded_list = self.train_list + self.test_list

        self.data = []
        self.targets = []
        self.filenames = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)

            with open(file_path, "rb") as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding="latin1")

                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                    self.filenames.extend(entry["filenames"])
                else:
                    self.targets.extend(entry["fine_labels"])
                    self.filenames.extend(entry["filenames"])

        self.data = np.vstack(self.data).reshape(-1, 1, 150, 150)
        self.data = self.data.transpose((0, 2, 3, 1))

        # Stratify entire data set according to input ratio (seeded)
        if test_size is not None:
            data_train, data_test, targets_train, targets_test = train_test_split(
                self.data,
                self.targets,
                test_size=test_size,
                stratify=self.targets,  # Targets to stratify according to
                random_state=42,
            )
            if self.train:
                self.data = data_train
                self.targets = targets_train
            else:
                self.data = data_test
                self.targets = targets_test

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError(
                "Dataset metadata file not found or corrupted."
                + " You can use download=True to download it"
            )
        with open(path, "rb") as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.reshape(img, (150, 150))

        if self.aug_type == "albumentations":
            if self.transform is not None:
                img = self.transform(image=img)["image"]

            if self.target_transform is not None:
                target = self.target_transform(image=target)["image"]

        elif self.aug_type == "torchvision":
            img = Image.fromarray(img, mode="L")
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            raise NotImplementedError(
                f"{self.aug_type} not implemented. Currently 'aug_type' must be either 'albumentations' which defaults to Albumentations or 'torchvision' to be functional."
            )

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            # print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        tmp = "train" if self.train is True else "test"
        fmt_str += "    Split: {}\n".format(tmp)
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str


# ---------------------------------------------------------------------------------


class MBFRFull(MiraBest_F):

    """
    Child class to load all FRI (0) & FRII (1)
    [100, 102, 104, 110, 112] and [200, 201, 210]
    """

    def __init__(self, *args, **kwargs):
        super(MBFRFull, self).__init__(*args, **kwargs)

        fr1_list = [0, 1, 2, 3, 4]
        fr2_list = [5, 6, 7]
        exclude_list = [8, 9]

        if exclude_list == []:
            return
        if self.train:
            targets = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            fr1 = np.array(fr1_list).reshape(1, -1)
            fr2 = np.array(fr2_list).reshape(1, -1)
            fr1_mask = (targets.reshape(-1, 1) == fr1).any(axis=1)
            fr2_mask = (targets.reshape(-1, 1) == fr2).any(axis=1)
            targets[fr1_mask] = 0  # set all FRI to Class~0
            targets[fr2_mask] = 1  # set all FRII to Class~1
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()
        else:
            targets = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            fr1 = np.array(fr1_list).reshape(1, -1)
            fr2 = np.array(fr2_list).reshape(1, -1)
            fr1_mask = (targets.reshape(-1, 1) == fr1).any(axis=1)
            fr2_mask = (targets.reshape(-1, 1) == fr2).any(axis=1)
            targets[fr1_mask] = 0  # set all FRI to Class~0
            targets[fr2_mask] = 1  # set all FRII to Class~1
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()


# ---------------------------------------------------------------------------------


class MBFRConfident(MiraBest_F):

    """
    Child class to load only confident FRI (0) & FRII (1)
    [100, 102, 104] and [200, 201]
    """

    def __init__(self, *args, **kwargs):
        super(MBFRConfident, self).__init__(*args, **kwargs)

        fr1_list = [0, 1, 2]
        fr2_list = [5, 6]
        exclude_list = [3, 4, 7, 8, 9]

        if exclude_list == []:
            return
        if self.train:
            targets = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            fr1 = np.array(fr1_list).reshape(1, -1)
            fr2 = np.array(fr2_list).reshape(1, -1)
            fr1_mask = (targets.reshape(-1, 1) == fr1).any(axis=1)
            fr2_mask = (targets.reshape(-1, 1) == fr2).any(axis=1)
            targets[fr1_mask] = 0  # set all FRI to Class~0
            targets[fr2_mask] = 1  # set all FRII to Class~1
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()
        else:
            targets = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            fr1 = np.array(fr1_list).reshape(1, -1)
            fr2 = np.array(fr2_list).reshape(1, -1)
            fr1_mask = (targets.reshape(-1, 1) == fr1).any(axis=1)
            fr2_mask = (targets.reshape(-1, 1) == fr2).any(axis=1)
            targets[fr1_mask] = 0  # set all FRI to Class~0
            targets[fr2_mask] = 1  # set all FRII to Class~1
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()


# ---------------------------------------------------------------------------------


class MBFRUncertain(MiraBest_F):

    """
    Child class to load only uncertain FRI (0) & FRII (1)
    [110, 112] and [210]
    """

    def __init__(self, *args, **kwargs):
        super(MBFRUncertain, self).__init__(*args, **kwargs)

        fr1_list = [3, 4]
        fr2_list = [7]
        exclude_list = [0, 1, 2, 5, 6, 8, 9]

        if exclude_list == []:
            return
        if self.train:
            targets = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            fr1 = np.array(fr1_list).reshape(1, -1)
            fr2 = np.array(fr2_list).reshape(1, -1)
            fr1_mask = (targets.reshape(-1, 1) == fr1).any(axis=1)
            fr2_mask = (targets.reshape(-1, 1) == fr2).any(axis=1)
            targets[fr1_mask] = 0  # set all FRI to Class~0
            targets[fr2_mask] = 1  # set all FRII to Class~1
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()
        else:
            targets = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            fr1 = np.array(fr1_list).reshape(1, -1)
            fr2 = np.array(fr2_list).reshape(1, -1)
            fr1_mask = (targets.reshape(-1, 1) == fr1).any(axis=1)
            fr2_mask = (targets.reshape(-1, 1) == fr2).any(axis=1)
            targets[fr1_mask] = 0  # set all FRI to Class~0
            targets[fr2_mask] = 1  # set all FRII to Class~1
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()


# ---------------------------------------------------------------------------------


class MBHybrid(MiraBest_F):

    """
    Child class to load confident(0) and uncertain (1) hybrid sources
    [110, 112] and [210]
    """

    def __init__(self, *args, **kwargs):
        super(MBHybrid, self).__init__(*args, **kwargs)

        h1_list = [8]
        h2_list = [9]
        exclude_list = [0, 1, 2, 3, 4, 5, 6, 7]

        if exclude_list == []:
            return
        if self.train:
            targets = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            h1 = np.array(h1_list).reshape(1, -1)
            h2 = np.array(h2_list).reshape(1, -1)
            h1_mask = (targets.reshape(-1, 1) == h1).any(axis=1)
            h2_mask = (targets.reshape(-1, 1) == h2).any(axis=1)
            targets[h1_mask] = 0  # set all FRI to Class~0
            targets[h2_mask] = 1  # set all FRII to Class~1
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()
        else:
            targets = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            h1 = np.array(h1_list).reshape(1, -1)
            h2 = np.array(h2_list).reshape(1, -1)
            h1_mask = (targets.reshape(-1, 1) == h1).any(axis=1)
            h2_mask = (targets.reshape(-1, 1) == h2).any(axis=1)
            targets[h1_mask] = 0  # set all FRI to Class~0
            targets[h2_mask] = 1  # set all FRII to Class~1
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()


# ---------------------------------------------------------------------------------


class MBRandom(MiraBest_F):

    """
    Child class to load 50 random FRI and 50 random FRII sources
    """

    def __init__(self, certainty="all", morphologies="all", *args, **kwargs):
        super(MBRandom, self).__init__(*args, **kwargs)

        # Checking flags
        # ------------------

        if certainty == "certain":
            certainty_list1 = np.array([0, 1, 2])
            certainty_list2 = np.array([5, 6])
        elif certainty == "uncertain":
            certainty_list1 = np.array([3, 4])
            certainty_list2 = np.array([7])
        else:
            certainty_list1 = np.array([0, 1, 2, 3, 4])
            certainty_list2 = np.array([5, 6, 7])

        if morphologies == "standard":
            morphology_list1 = np.array([0, 3])
            morphology_list2 = np.array([5, 7])
        else:
            morphology_list1 = np.array([0, 1, 2, 3, 4])
            morphology_list2 = np.array([5, 6, 7])

        list_matches1 = np.in1d(certainty_list1, morphology_list1)
        list_matches2 = np.in1d(certainty_list2, morphology_list2)

        h1_list = certainty_list1[np.where(list_matches1)[0]]
        h2_list = certainty_list2[np.where(list_matches2)[0]]

        # ------------------

        if self.train:
            targets = np.array(self.targets)
            h1 = np.array(h1_list).reshape(1, -1)
            h2 = np.array(h2_list).reshape(1, -1)
            h1_mask = (targets.reshape(-1, 1) == h1).any(axis=1)
            h2_mask = (targets.reshape(-1, 1) == h2).any(axis=1)
            h1_indices = np.where(h1_mask)
            h2_indices = np.where(h2_mask)
            h1_random = np.random.choice(h1_indices[0], 50, replace=False)
            h2_random = np.random.choice(h2_indices[0], 50, replace=False)
            targets[h1_random] = 0  # set all FRI to Class~0
            targets[h2_random] = 1  # set all FRII to Class~1
            target_list = np.concatenate((h1_random, h2_random))
            exclude_mask = (targets.reshape(-1, 1) == target_list).any(axis=1)
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()
        else:
            targets = np.array(self.targets)
            h1 = np.array(h1_list).reshape(1, -1)
            h2 = np.array(h2_list).reshape(1, -1)
            h1_mask = (targets.reshape(-1, 1) == h1).any(axis=1)
            h2_mask = (targets.reshape(-1, 1) == h2).any(axis=1)
            h1_indices = np.where(h1_mask)
            h2_indices = np.where(h2_mask)
            h1_random = np.random.choice(h1_indices[0], 50, replace=False)
            h2_random = np.random.choice(h2_indices[0], 50, replace=False)
            targets[h1_random] = 0  # set all FRI to Class~0
            targets[h2_random] = 1  # set all FRII to Class~1
            target_list = np.concatenate((h1_random, h2_random))
            exclude_mask = (targets.reshape(-1, 1) == target_list).any(axis=1)
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()
